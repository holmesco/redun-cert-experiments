from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple
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
    # Verbosity flag for debugging
    verbose: bool = False

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
    # inlier threshold for RANSAC pose estimation (in meters)
    # NOTE: This should be set to the same value as the 0.5x invariant_epsilon parameter for RANSAC to generate cliques of the graph.
    ransac_inlier_threshold: float = 0.15


class DataAssociationBlock:
    """Data association block that takes in two sets of 3D keypoints and outputs a set of matched keypoints.

    Uses CLIPPER for 3D data association: takes in two sets of 3D keypoints and outputs a set of matched keypoints.
    """

    def __init__(self, config: DataAssociationConfig):
        # Store config
        self.config = config
        # Check RANSAC parameters
        if self.config.method == DataAssociationMethod.RANSAC:
            assert (
                self.config.ransac_num_sample_pts >= 3
            ), "RANSAC requires at least 3 points for pose estimation."
            assert (
                self.config.ransac_num_iterations > 0
            ), "RANSAC requires at least 1 iteration."
            assert (
                self.config.ransac_inlier_threshold > 0
            ), "RANSAC requires a positive inlier threshold."
            if (
                self.config.ransac_inlier_threshold
                > 0.5 * self.config.invariant_epsilon
            ):
                print(
                    "Warning: RANSAC inlier threshold is greater than 0.5x invariant epsilon. This may lead to RANSAC inliers that are not cliques of the data association graph."
                )

        # Affinity matrix
        self.M: np.ndarray | None = None
        self.M_torch: torch.Tensor | None = None
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
        x: np.ndarray,
        cost: float = None,
        check_constraints: bool = False,
    ) -> Tuple[AnalyticCenterResult, float]:
        """Certify the solution x for the max clique problem defined by M.

        Parameters
        ----------
        x : np.ndarray
            Solution vector for the max clique problem, of shape (N,).
        cost : float, optional
            Cost of the solution, defined as -x^T M x. If not provided, it will be computed from x and M.
        check_constraints : bool, optional
            If True, check that the solution x satisfies the constraints of the max clique problem. Default

        Returns
        -------
        result : AnalyticCenterResult
            Result of the certification process.
        """
        # Retrieve the affinity matrix from the last forward pass
        M = self.get_affinity()
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
        return result, cost

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
            soln (torch.Tensor): Solution vector for the max clique problem, of shape (N,).
            cost (float): Cost of the solution, defined as -x^T M x.
        """
        if self.M is None:
            raise ValueError("Affinity matrix has not been set up.")

        device = inliers.device
        M = self.get_affinity(use_torch=True, device=device)

        # Select submatrix of M corresponding to inliers
        inlier_idx = torch.nonzero(inliers.to(device).bool(), as_tuple=True)[0]
        M_sub = M[inlier_idx][:, inlier_idx]

        if not torch.all(M_sub > 0):
            if self.config.verbose:
                print("Cost submatrix contains non-positive elements. Inliers do not form a clique.")
            return None, float("inf")        

        # Power iteration to get Perron vector
        v = torch.ones(len(inlier_idx), dtype=M.dtype, device=device)
        v /= torch.linalg.norm(v)
        for i in range(self.config.clique_to_solution_iters):
            v_new = M_sub @ v
            v_new /= torch.linalg.norm(v_new)
            if torch.linalg.norm(v_new - v) < self.config.clique_to_solution_tol:
                break
            v = v_new
        v = v_new
        cost = -float(v @ M_sub @ v)

        # Embed Perron vector into full solution
        soln = torch.zeros(M.shape[0], dtype=M.dtype, device=device)
        soln[inlier_idx] = v
        return soln, cost

    def get_affinity(self, use_torch=False, device=torch.device("cpu")):
        """Get the affinity matrix from the CLIPPER block.
        Warning: This should only be called after set_up_affinity_matrix has been called, otherwise the affinity matrix will be None.
        Returns:
            M (np.ndarray): Affinity matrix, of shape (N, N).
        """
        if self.M is None:
            raise ValueError(
                "Affinity matrix has not been set up. Call set_up_affinity_matrix() first."
            )
        if use_torch:
            if self.M_torch is None:
                self.M_torch = torch.from_numpy(self.M).to(device)
            return torch.from_numpy(self.M).to(device)
        return self.M

    def run_ransac(
        self,
        kpt_3D_src: torch.Tensor,
        kpt_3D_trg: torch.Tensor,
    ):
        """Forward pass through the RANSAC block. Keypoints with same index are assumed to be putative correspondences. The RANSAC block will output a mask of inliers for the matched keypoints.
        Args:
            kpt_3D_src (torch.Tensor): Source 3D keypoints, of shape (4, N).
            kpt_3D_trg (torch.Tensor): Target 3D keypoints, of shape (4, N).
        Returns:
            inliers (torch.Tensor): Inlier mask for the matched keypoints, of shape (N,).
        """
        # Set up affinity matrix for max clique problem.
        self.set_up_affinity_matrix(kpt_3D_src, kpt_3D_trg)
        # track best inliers and cost
        best_inliers = None
        best_cost = np.inf
        best_soln = None
        with torch.no_grad():
            for i in range(self.config.ransac_num_iterations):
                # Randomly sample 3 points
                idx = torch.randperm(kpt_3D_src.shape[1])[
                    : self.config.ransac_num_sample_pts
                ]
                src_sample = kpt_3D_src[:, idx]
                trg_sample = kpt_3D_trg[:, idx]

                # Estimate pose using SVD
                T_trg_src = estimate_pose_svd(
                    src_sample,
                    trg_sample,
                )
                # Transform source points to target frame
                kpt_3D_src_inTrg = T_trg_src @ kpt_3D_src
                # Compute distances between transformed source points and target points
                distances = torch.norm(
                    kpt_3D_src_inTrg[:3, :] - kpt_3D_trg[:3, :], dim=0
                )
                # Determine inliers based on distance threshold
                inliers = distances < self.config.ransac_inlier_threshold
                # Convert RANSAC inliers to solution vector for the max clique problem
                soln, cost = self.inliers_to_solution(inliers)
                # Update best inliers if cost is better
                if cost < best_cost or best_inliers is None:
                    best_cost = cost
                    best_inliers = inliers
                    best_soln = soln
        best_soln = best_soln.cpu().numpy() if best_soln is not None else None
        return best_inliers, best_soln, best_cost


def estimate_pose_svd(keypoints_3D_src, keypoints_3D_trg):
    """Estimate the relative pose between the source and target frames using SVD.

    Args:
        keypoints_3D_src (torch.Tensor): 3D point coordinates of keypoints from source frame, of shape (4, N).
        keypoints_3D_trg (torch.Tensor): 3D point coordinates of keypoints from target frame, of shape (4, N).
    Returns:
        T_trg_src (torch.Tensor): Relative transform from the source to the target frame,
    """

    # Compute centroids (elementwise multiplication/division)
    n_points = keypoints_3D_src.shape[1]
    centroid_src = (
        torch.sum(keypoints_3D_src[0:3, :], dim=1, keepdim=True) / n_points
    )  # 3x1
    centroid_trg = (
        torch.sum(keypoints_3D_trg[0:3, :], dim=1, keepdim=True) / n_points
    )  # 3x1
    # Compute centered coordinates
    src_centered = keypoints_3D_src[0:3, :] - centroid_src  # 3xN
    trg_centered = keypoints_3D_trg[0:3, :] - centroid_trg
    # Compute rotation and translation (T_trg_src in sensor frame)
    H = trg_centered @ src_centered.transpose(1, 0).contiguous()  # 3x3
    U, S, V = torch.svd(H)
    det_UV = torch.det(U) * torch.det(V)
    diag = torch.diag_embed(torch.Tensor([1.0, 1.0, det_UV]).type_as(V))  # 3x3
    R_trg_src = U @ diag @ V.transpose(1, 0)  # 3x3
    # Translation from trg to src given in src frame
    # NOTE: Uses the fact that the centroids should be coincident after rotation.
    t_src_trg_intrg = centroid_trg - R_trg_src @ centroid_src  # 3x1

    # Create translation matrix
    zeros = torch.zeros(1, 3).type_as(V)  # 1x3
    one = torch.ones(1, 1).type_as(V)  # 1x1
    trans_cols = torch.cat([t_src_trg_intrg, one], dim=0)  # 4x1
    rot_cols = torch.cat([R_trg_src, zeros], dim=0)  # 4x3
    T_trg_src = torch.cat([rot_cols, trans_cols], dim=1)  # 4x4

    return T_trg_src


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
