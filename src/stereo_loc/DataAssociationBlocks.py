from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
from scipy.sparse import csc_array, coo_array, eye_array
import clipperpy

from stereo_loc.AnalyticCenterParamsConfig import AnalyticCenterParamsConfig
from ranktools import AnalyticCenterResult, AnalyticCenter, MaxCliqueCertifier
from cert_tools.sdp_solvers import solve_sdp_fusion


class DataAssociationMethod(Enum):
    CLIPPER = "CLIPPER"
    RANSAC = "RANSAC"
    CLIPPER_SDP = "CLIPPER_SDP"


@dataclass
class DataAssociationConfig:
    """Configuration for the CLIPPER data association module."""

    # Method for data association. Options: "clipper", "ransac"
    method: DataAssociationMethod = DataAssociationMethod.CLIPPER
    # Device
    default_device: str = "cpu"
    
    # --- Graph Definition Parameters ---   
    invariant_epsilon: float = 0.3  # 30 cm, correspoding to max allowable discrepancy
    invariant_sigma: float = (
        0.15  # 15 cm, half the max allowable discrepancy, to get a good scoring function
    )
    # Threshold for converting to unweighted graph. (zero if not converting)
    unweighted: bool = False
    
    # CLIPPER rounding strategy
    clipper_rounding_method = clipperpy.Rounding.DSD_HEU
    
    # --- Certification Parameters ---
    # Certification flag for data association
    certify: bool = False
    # Parameters for the analytic centering certifier
    ac_params: AnalyticCenterParamsConfig = field(
        default_factory=AnalyticCenterParamsConfig
    )
    # Rank ratio for determining rank of SDP solution. Eigenvalue considered to be zero if it is less than rank_ratio * max_eigenvalue. This is used to determine if the SDP solution is rank-1.
    rank_ratio: float = 1e-6
    
    # --- Clique to Solution Conversion Parameters ---
    # inlier to solution conversion iterations
    clique_to_solution_iters: int = 100
    # inlier to solution tolerance
    clique_to_solution_tol: float = 1e-9
    
    # --- RANSAC parameters ---
    # number of points to use for RANSAC pose estimation
    ransac_num_sample_pts: int = 3
    # number of RANSAC iterations
    ransac_num_iterations: int = 50
    
class DataAssociationBlock:
    """Data association block that takes in two sets of 3D keypoints and outputs a set of matched keypoints.

    Uses CLIPPER for 3D data association: takes in two sets of 3D keypoints and outputs a set of matched keypoints."""

    def __init__(self, config: DataAssociationConfig):
        self.config = config
        # Affinity matrix
        self.M: np.ndarray | None = None
        # Set up invariant
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma = self.config.invariant_sigma
        iparams.epsilon = self.config.invariant_epsilon
        invariant = clipperpy.invariants.EuclideanDistance(iparams)
        # Define rounding strategy
        params = clipperpy.Params()
        params.rounding = self.config.clipper_rounding_method
        # define clipper object
        self.clipper = clipperpy.CLIPPER(invariant, params)

    def certify_solution(
        self,
        soln: np.ndarray | None = None,
        inliers: torch.Tensor | None = None,
        cost: float = None,
        check_constraints: bool = False,
    ) -> AnalyticCenterResult:
        """Certify the solution x for the max clique problem defined by M.

        Parameters
        ----------
        inliers : np.ndarray
            Solution vector to certify.

        Returns
        -------
        result : AnalyticCenterResult
            Result of the certification process.
        """
        assert (
            soln is not None or inliers is not None
        ), "Either x or inliers must be provided."
        # Retrieve the affinity matrix from the last forward pass
        M = self.get_affinity()

        # Convert inliers to feasible solution if they are provided
        if inliers is not None:
            inliers_np = inliers.cpu().numpy()[:, None]
            x = inliers_np / np.linalg.norm(inliers_np)
        else:
            x = soln
        # Check constraints
        if check_constraints:
            # Get the constraints for the max clique problem
            constraints, values = get_maxclique_sdp_constraints(M)
            for i, (A, b) in enumerate(zip(constraints, values)):
                assert np.abs(x.T @ A @ x - b) < 1e-10, f"Constraint {i} violated!"
        # Get the cost of the solution if not provided
        if cost is None:
            cost = -(x.T @ M @ x).item()
        # Set up central path certifier
        ac_params = self.config.ac_params.to_cpp_class()
        certifier = MaxCliqueCertifier(-M, cost, ac_params)
        # Certify the solution
        result = certifier.certify(x)
        return result

    def set_up_affinity_matrix(self, kpt_3D_src, kpt_3D_trg):
        """Set up the affinity matrix for the CLIPPER block. This is a separate function to allow for reusing the affinity matrix for certification.
        Args:
            kpt_3D_src (torch.Tensor): Source 3D keypoints, of shape (4, N).
            kpt_3D_trg (torch.Tensor): Target 3D keypoints, of shape (4, N).
        """
        # Convert to numpy arrays
        kpt_3D_src_np = kpt_3D_src[:3, :].double().cpu().numpy()  # (3,N), float64
        kpt_3D_trg_np = kpt_3D_trg[:3, :].double().cpu().numpy()  # (3,N), float64
        N = kpt_3D_src_np.shape[1]  # Num associations
        # Putative correspondences
        A = np.zeros((N, 2), dtype=np.int32)
        for i in range(N):
            A[i, 0] = i
            A[i, 1] = i
        # Get pairwise consistency matrix
        self.clipper.score_pairwise_consistency(kpt_3D_src_np, kpt_3D_trg_np, A)
        # Get matrix
        self.M = self.clipper.get_affinity_matrix()
        # thresholding to get unweighted graph if enabled
        if self.config.unweighted:
            self.M = (self.M > 0.0).astype(float)
            # Set constraint and affinity matrix to thresholded values.
            self.clipper.set_matrix_data(M=self.M, C=self.M)

    def run_clipper(
        self,
        kpt_3D_src: torch.Tensor | None = None,
        kpt_3D_trg: torch.Tensor | None = None,
    ):
        """Forward pass through the CLIPPER block. Keypoints with same index are assumed to be putative correspondences. The CLIPPER block will output a mask of inliers for the matched keypoints.
        Args:
            kpt_3D_src (torch.Tensor): Source 3D keypoints, of shape (4, N).
            kpt_3D_trg (torch.Tensor): Target 3D keypoints, of shape (4, N).
        Returns:
            inliers (torch.Tensor): Inlier mask for the matched keypoints, of shape (N,).
        """
        # Set up affinity matrix for max clique problem.
        if kpt_3D_src is not None and kpt_3D_trg is not None:
            self.set_up_affinity_matrix(kpt_3D_src, kpt_3D_trg)
        elif self.M is None:
            raise ValueError(
                "Affinity matrix has not been set up. Provide keypoints or call set_up_affinity_matrix first."
            )
        # Run CLIPPER
        self.clipper.solve()
        # retrieve inliers
        soln = self.clipper.get_solution()
        thresh = np.max(soln.u) / 2
        inliers = torch.from_numpy(soln.u > thresh).bool()  # (N,)
        return inliers, soln.u

    def run_sdp(self, kpt_3D_src, kpt_3D_trg):
        """Run the SDP relaxation of the max clique problem defined by the affinity matrix M. This is a separate function to allow for reusing the affinity matrix for certification."""
        # Set up affinity matrix for max clique problem.
        if kpt_3D_src is not None and kpt_3D_trg is not None:
            self.set_up_affinity_matrix(kpt_3D_src, kpt_3D_trg)
        elif self.M is None:
            raise ValueError(
                "Affinity matrix has not been set up. Provide keypoints or call set_up_affinity_matrix first."
            )
        # Get the constraints for the max clique problem
        As, bs = get_maxclique_sdp_constraints(self.M)
        constraints = [(A, b) for A, b in zip(As, bs)]
        # Solve SDP: min <Q, X> s.t. <A_i, X> = b_i, X >= 0
        X_sol, info = solve_sdp_fusion(
            Q=-self.M,
            Constraints=constraints,
            adjust=False,
            verbose=True,
        )
        time_sdp = info["time"]
        print(f"SDP solve time: {time_sdp*1e3:.0f} ms")

        # Extract rank-1 solution via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(X_sol)
        # Determine the rank based on relative ratio of eigenvalues
        max_eigval = eigvals[-1]
        rank = np.sum(eigvals > self.config.rank_ratio * max_eigval)
        if rank > 1:
            print(f"Warning: SDP solution is not rank-1. Rank: {rank}")
        # Leading eigenvector (largest eigenvalue)
        U = eigvecs[:, -rank:] * np.sqrt(np.maximum(eigvals[-rank:], 0.0))
        # Convert to inlier mask
        inliers = None
        if rank == 1:
            thresh = np.max(U) / 2
            inliers = torch.from_numpy(U[:, 0] > thresh).bool()  # (N,)
        return inliers, U

    def inliers_to_solution(self, inliers: torch.Tensor):
        """Convert inlier mask to solution vector for the max clique problem defined by M.
        This is done by identfying the max eigenvector of the submatrix of M corresponding to the inliers, and embedding it into the full solution vector.
        Args:
            inliers (torch.Tensor): Inlier mask for the matched keypoints, of shape (N,).
        Returns:
            soln (np.ndarray): Solution vector for the max clique problem, of shape (N,).
            cost (float): Cost of the solution, defined as -x^T M x.
        """
        if self.M is None:
            raise ValueError(
                "Affinity matrix has not been set up. Call forward() first."
            )

        # Select submatrix of M corresponding to inliers
        inlier_idx = np.where(inliers.cpu().numpy())[0]
        M_sub = self.M[np.ix_(inlier_idx, inlier_idx)]

        assert np.all(
            M_sub > 0
        ), "Cost submatrix contains non-positive elements. Inliers do not form a clique."

        # Power iteration to get Perron vector
        v = np.ones(len(inlier_idx))
        v /= np.linalg.norm(v)
        for i in range(self.config.clique_to_solution_iters):
            v_new = M_sub @ v
            v_new /= np.linalg.norm(v_new)
            if np.linalg.norm(v_new - v) < self.config.clique_to_solution_tol:
                break
            v = v_new
        v = v_new
        cost = -float(v @ M_sub @ v)

        # Embed Perron vector into full solution
        soln = np.zeros(self.M.shape[0])
        soln[inlier_idx] = v
        return soln, cost

    def get_affinity(self):
        """Get the affinity matrix from the CLIPPER block.
        Warning: This should only be called after the forward pass, and will return the affinity matrix for the last pair of keypoints that were passed through the forward pass.
        Returns:
            M (np.ndarray): Affinity matrix, of shape (N, N).
        """
        if self.M is None:
            raise ValueError(
                "Affinity matrix has not been set up. Call forward() first."
            )
        return self.M

    def run_ransac(
        self,
        kpt_3D_src: torch.Tensor | None = None,
        kpt_3D_trg: torch.Tensor | None = None,
    ):
        """Forward pass through the RANSAC block. Keypoints with same index are assumed to be putative correspondences. The RANSAC block will output a mask of inliers for the matched keypoints.
        Args:
            kpt_3D_src (torch.Tensor): Source 3D keypoints, of shape (4, N).
            kpt_3D_trg (torch.Tensor): Target 3D keypoints, of shape (4, N).
        Returns:
            inliers (torch.Tensor): Inlier mask for the matched keypoints, of shape (N,).
        """
        raise NotImplementedError(
            "RANSAC not implemented for base DataAssociationBlock class."
        )


def get_maxclique_sdp_constraints(M: np.ndarray):
    """Optimized version using vectorized index extraction and fast COO construction.
    NOTE: This function is only used for running the SDP relaxation but it is quite slow.
    TODO: Implement in MOSEK call in C++"""
    n = M.shape[0]

    # 1. Get upper triangle indices where M == 0 efficiently
    iu_rows, iu_cols = np.triu_indices_from(M, k=1)
    non_edge_mask = M[iu_rows, iu_cols] == 0
    rows = iu_rows[non_edge_mask]
    cols = iu_cols[non_edge_mask]

    num_non_edges = len(rows)

    # 2. Fast generation of symmetric sparse constraints
    # Reusing the same data and shape allocations minimizes overhead
    ones = np.ones(2, dtype=np.float64)
    shape = (n, n)

    # Construct COO arrays first (fastest for initialization), then convert to CSC
    constraints = [
        coo_array((ones, ([r, c], [c, r])), shape=shape).tocsc()
        for r, c in zip(rows, cols)
    ]

    # 3. Add the trace constraint (Identity matrix)
    # Using eye_array is faster and cleaner than csc_array(np.eye(n))
    constraints.append(eye_array(n, format="csc"))

    # 4. Preallocate values array directly
    values = np.zeros(num_non_edges + 1, dtype=np.float64)
    values[-1] = 1.0  # Trace constraint

    return constraints, values
