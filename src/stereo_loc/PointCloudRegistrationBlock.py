from dataclasses import dataclass, field
from typing import Tuple

import torch
import numpy as np

from ranktools import (
    CPCertParams,
    LinearSolverType,
    LowRankPrecondMethod,
    CPCertResult,
)
from stereo_loc.CPCertParamsConfig import CPCertParamsConfig
from mat_weight_loc.one_pose_stereo_loc import SinglePoseStereoLocalization
import gtsam


@dataclass
class PointCloudRegistrationConfig:
    """Configuration for the point cloud registration block."""

    # Certification flag
    certify: bool = False
    # CPCert parameters
    cpcert_params: CPCertParamsConfig = field(
        default_factory=CPCertParamsConfig
    )
    # Verbosity flag for factor graph optimization
    verbose: bool = False
    # Cost adjustment  flag for certification
    adjust_cost: bool = False


class PointCloudRegistrationBlock:
    """Wrapper class for point cloud registration blocks. Takes in two sets of 3D keypoints and outputs the relative transform between them."""

    def __init__(
        self,
        config: PointCloudRegistrationConfig,
        keypoints_3D_src: torch.Tensor,
        keypoints_3D_trg: torch.Tensor,
        inv_cov_weights: torch.Tensor,
    ):
        """Initialize the point cloud registration block."""
        self.config = config
        # Move to cpu and switch to double precision
        # NOTE: single precision was causing some issues with symmetry checks.
        self.keypoints_3D_src = keypoints_3D_src.to("cpu").float().numpy()
        self.keypoints_3D_trg = keypoints_3D_trg.to("cpu").float().numpy()
        self.inv_cov_weights = inv_cov_weights.to("cpu").float().numpy()

        # Create instance of the SinglePoseStereoLocalization class
        self.localizer = SinglePoseStereoLocalization(
            keypoints_3D_src=self.keypoints_3D_src,
            keypoints_3D_trg=self.keypoints_3D_trg,
            inv_cov_weights=self.inv_cov_weights,
            certify=self.config.certify,
            T_s_v=None,
        )

    def solve_factor_graph(
        self, T_init: np.ndarray, verbose: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """Solve the factor graph optimization problem starting from T_init.
        T_init is assumed to be the transform from the target to source frames, T_src_trg.
        """
        # Invert T_init to get the transform from source to target frames, T_trg_src
        T_trg_src_init = np.linalg.inv(T_init)
        T_trg_src, info = self.localizer.solve_factor_graph(
            T_init=T_trg_src_init, verbose=verbose
        )
        # reinvert T_trg_src to get T_src_trg
        T_src_trg = np.linalg.inv(T_trg_src)
        return T_src_trg, info

    def certify_solution(self, T_src_trg: np.ndarray) -> CPCertResult:
        """Certify the solution using the CPCert certificate."""
        if not self.config.certify:
            raise ValueError("Certification is not enabled in the configuration.")
        # Invert solution to get T_trg_src for certification
        T_trg_src = np.linalg.inv(T_src_trg)
        # Convert dataclass params to C++ wrapper config class
        cpcert_params: CPCertParams = self.config.cpcert_params.to_cpp_class()
        # Set certifier parameters
        self.localizer.set_certifier_params(cpcert_params)
        # Run certification
        result = self.localizer.certify_single_pose_solution(
            T_est=T_trg_src,
            verbose=self.config.verbose,
            adjust_cost=self.config.adjust_cost,
        )
        return result

    def solve_sdp(self, verbose: bool = False) -> Tuple[np.ndarray, dict]:
        """Solve the SDP relaxation of the registration problem."""
        X, info = self.localizer.solve_sdp(verbose=verbose)
        # Check rank-1 solution
        eigenvalues = np.linalg.eigvalsh(X)
        eigenvalues = np.sort(eigenvalues)[::-1]
        assert (
            eigenvalues[0] / eigenvalues[1] > 1e6
        ), f"Eigenvalue ratio: {eigenvalues[0] / eigenvalues[1]}"
        # Check that solution is close to ground truth
        values = self.localizer.values_from_vector(X[:, 0])
        key = gtsam.Symbol("x", 0).key()
        T_trg_src_sdp = values.atPose3(key)
        # Update cost from solution
        info["cost"] = self.localizer.graph.error(values)
        return T_trg_src_sdp, info


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
