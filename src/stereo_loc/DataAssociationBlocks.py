from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
from scipy.sparse import csc_array

import clipperpy

from stereo_loc.AnalyticCenterParamsConfig import AnalyticCenterParamsConfig
from ranktools import AnalyticCenterResult, AnalyticCenter
from cert_tools.sdp_solvers import solve_sdp_fusion


class DataAssociationMethod(Enum):
    CLIPPER = "CLIPPER"
    RANSAC = "RANSAC"
    CLIPPER_SDP = "CLIPPER_SDP"


@dataclass
class ClipperConfig:
    """Configuration for the CLIPPER data association module."""

    # CLIPPER invariant parameters
    invariant_epsilon: float = 0.3  # 30 cm, correspoding to max allowable discrepancy
    invariant_sigma: float = (
        0.15  # 15 cm, half the max allowable discrepancy, to get a good scoring function
    )
    # CLIPPER rounding strategy
    clipper_rounding_method = clipperpy.Rounding.DSD_HEU
    # Threshold for converting to unweighted graph. (zero if not converting)
    threshold: bool = False


@dataclass
class DataAssociationConfig:
    """Configuration for the CLIPPER data association module."""

    clipper_config: ClipperConfig = field(default_factory=ClipperConfig)
    # Certification flag for data association
    certify: bool = False
    # Parameters for the analytic centering certifier
    ac_params: AnalyticCenterParamsConfig = field(
        default_factory=AnalyticCenterParamsConfig
    )


class DataAssociationBlock:
    """Data association block that takes in two sets of 3D keypoints and outputs a set of matched keypoints."""

    def __init__(self, config: DataAssociationConfig):
        self.config = config

    def forward(self, kpt_3D_src, kpt_3D_trg) -> torch.Tensor:
        """Forward pass through the data association block.
        Args:
            kpt_3D_src (torch.Tensor): Source 3D keypoints, of shape (4, N).
            kpt_3D_trg (torch.Tensor): Target 3D keypoints, of shape (4, N).
        Returns:
            inliers (torch.Tensor): Inlier mask for the matched keypoints, of shape (N,).
        """
        raise NotImplementedError(
            "Forward pass not implemented for base DataAssociationBlock class."
        )

    def get_affinity(self) -> np.ndarray:
        """Get the affinity matrix from the data association block.
        Warning: This should only be called after the forward pass, and will return the affinity matrix for the last pair of keypoints that were passed through the forward pass.
        Returns:
            M (np.ndarray): Affinity matrix, of shape (N, N).
        """
        raise NotImplementedError(
            "Get affinity not implemented for base DataAssociationBlock class."
        )

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
        # Get the constraints for the max clique problem
        constraints, values = get_maxclique_sdp_constraints(M)
        # Convert inliers to feasible solution if they are provided
        if inliers is not None:
            inliers_np = inliers.cpu().numpy()[:, None]
            x = inliers_np / np.linalg.norm(inliers_np)
        else:
            x = soln
        # Check constraints
        if check_constraints:
            for i, (A, b) in enumerate(zip(constraints, values)):
                assert np.abs(x.T @ A @ x - b) < 1e-10, f"Constraint {i} violated!"
        # Get the cost of the solution if not provided
        if cost is None:
            cost = -(x.T @ M @ x).item()
        # Set up central path certifier
        ac_params = self.config.ac_params.to_cpp_class()
        certifier = AnalyticCenter(-M, cost, constraints, values, ac_params)
        # Certify the solution
        result = certifier.certify(x)
        return result


class ClipperBlock(DataAssociationBlock):
    """CLIPPER block for 3D data association. Takes in two sets of 3D keypoints and outputs a set of matched keypoints."""

    def __init__(self, config: DataAssociationConfig):
        super().__init__(config)
        # Set up invariant
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma = self.config.clipper_config.invariant_sigma
        iparams.epsilon = self.config.clipper_config.invariant_epsilon
        invariant = clipperpy.invariants.EuclideanDistance(iparams)
        # Define rounding strategy
        params = clipperpy.Params()
        params.rounding = self.config.clipper_config.clipper_rounding_method
        # define clipper object
        self.clipper = clipperpy.CLIPPER(invariant, params)

    def forward(self, kpt_3D_src, kpt_3D_trg):
        """Forward pass through the CLIPPER block. Keypoints with same index are assumed to be putative correspondences. The CLIPPER block will output a mask of inliers for the matched keypoints.
        Args:
            kpt_3D_src (torch.Tensor): Source 3D keypoints, of shape (4, N).
            kpt_3D_trg (torch.Tensor): Target 3D keypoints, of shape (4, N).
        Returns:
            inliers (torch.Tensor): Inlier mask for the matched keypoints, of shape (N,).
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
        # thresholding to get unweighted graph if enabled
        if self.config.clipper_config.threshold:
            M = self.clipper.get_affinity_matrix()
            M = (M > 0.0).astype(float)
            # Set constraint and affinity matrix to thresholded values.
            self.clipper.set_matrix_data(M=M, C=M)
        # Run CLIPPER
        self.clipper.solve()
        # retrieve inliers
        soln = self.clipper.get_solution()
        thresh = np.max(soln.u) / 2
        inliers = torch.from_numpy(soln.u > thresh).bool()  # (N,)
        return inliers, soln.u

    def get_affinity(self):
        """Get the affinity matrix from the CLIPPER block.
        Warning: This should only be called after the forward pass, and will return the affinity matrix for the last pair of keypoints that were passed through the forward pass.
        Returns:
            M (torch.Tensor): Affinity matrix, of shape (N, N).
        """
        return self.clipper.get_affinity_matrix()


class ClipperSDPBlock(DataAssociationBlock):
    """CLIPPER SDP block for 3D data association. Takes in two sets of 3D keypoints and outputs a set of matched keypoints."""

    def __init__(self, config: DataAssociationConfig):
        super().__init__(config)
        # Set up invariant
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma = self.config.clipper_config.invariant_sigma
        iparams.epsilon = self.config.clipper_config.invariant_epsilon
        invariant = clipperpy.invariants.EuclideanDistance(iparams)
        # Define rounding strategy
        params = clipperpy.Params()
        params.rounding = self.config.clipper_config.clipper_rounding_method
        # define clipper object
        self.clipper = clipperpy.CLIPPER(invariant, params)

    def forward(self, kpt_3D_src, kpt_3D_trg):
        """Forward pass through the CLIPPER SDP block. Keypoints with same index are assumed to be putative correspondences. The CLIPPER block will output a mask of inliers for the matched keypoints.
        Args:
            kpt_3D_src (torch.Tensor): Source 3D keypoints, of shape (4, N).
            kpt_3D_trg (torch.Tensor): Target 3D keypoints, of shape (4, N).
        Returns:
            inliers (torch.Tensor): Inlier mask for the matched keypoints, of shape (N,).
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
        # thresholding to get unweighted graph if enabled
        M = self.clipper.get_affinity_matrix()
        if self.config.clipper_config.threshold:
            M = (M > 0.0).astype(float)
        # Get the constraints for the max clique problem
        As, bs = get_maxclique_sdp_constraints(M)
        constraints = [(A, b) for A, b in zip(As, bs)]
        # Solve SDP: min <Q, X> s.t. <A_i, X> = b_i, X >= 0
        X_sol, info = solve_sdp_fusion(
            Q=-M,
            Constraints=constraints,
            adjust=False,
            verbose=True,
        )
        time_sdp = info["time"]
        print(f"SDP solve time: {time_sdp*1e3:.0f} ms")

        # # Extract rank-1 solution via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(X_sol)
        # Leading eigenvector (largest eigenvalue)
        u = eigvecs[:, -1] * np.sqrt(np.maximum(eigvals[-1], 0.0))
        # Convert to inlier mask
        thresh = np.max(u) / 2
        inliers = torch.from_numpy(u > thresh).bool()  # (N,)
        return inliers, u

    def get_affinity(self):
        """Get the affinity matrix from the CLIPPER block.
        Warning: This should only be called after the forward pass, and will return the affinity matrix for the last pair of keypoints that were passed through the forward pass.
        Returns:
            M (torch.Tensor): Affinity matrix, of shape (N, N).
        """
        return self.clipper.get_affinity_matrix()


def get_maxclique_sdp_constraints(M: np.ndarray):
    """Get the constraints of the maximum clique problem.

    Parameters
    ----------
    M : np.ndarray
        Affinity matrix of the problem.

    Returns
    -------
    constraints : list of scipy.sparse.csc_array
        List of sparse matrices representing the constraints of the problem.
    values : np.ndarray
        Values corresponding to the constraints (e.g., 0 for non-edges, 1 for trace constraint).
    """
    # Find indices where M is zero and j > i
    rows, cols = np.where((M == 0) & (np.triu(np.ones(M.shape, dtype=bool), k=1)))
    constraints = [
        csc_array(([1.0, 1.0], ([r, c], [c, r])), shape=M.shape)
        for r, c in zip(rows, cols)
    ]

    # Preallocate values array
    values = np.zeros(len(constraints) + 1)
    values[:-1] = 0.0  # Non-edges
    values[-1] = 1.0  # Trace constraint

    # Add the trace constraint
    sparse_identity = csc_array(np.eye(M.shape[0]))
    constraints.append(sparse_identity)

    return constraints, values
