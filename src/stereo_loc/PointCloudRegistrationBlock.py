from dataclasses import dataclass
from typing import Tuple

import torch
import numpy as np

from ranktools import (
    AnalyticCenterParams,
    LinearSolverType,
    LowRankPrecondMethod,
    AnalyticCenterResult,
)
from mat_weight_loc.one_pose_stereo_loc import SinglePoseStereoLocalization


@dataclass
class PointCloudRegistrationConfig:
    """Configuration for the point cloud registration block."""

    # Certification flag
    certify: bool = False
    # Analytic centering parameters
    ac_params: AnalyticCenterParams = AnalyticCenterParams()
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
        T_init is assumed to be the transform from the target to source frames, T_src_trg."""
        # Invert T_init to get the transform from source to target frames, T_trg_src
        T_trg_src_init = np.linalg.inv(T_init)
        T_trg_src, info = self.localizer.solve_factor_graph(T_init=T_trg_src_init, verbose=verbose)
        # reinvert T_trg_src to get T_src_trg
        T_src_trg = np.linalg.inv(T_trg_src)
        return T_src_trg, info

    def certify_single_pose_solution(self, T_est: torch.Tensor) -> AnalyticCenterResult:
        """Certify the solution using the analytic center certificate."""
        if not self.config.certify:
            raise ValueError("Certification is not enabled in the configuration.")
        # Set certifier parameters
        self.localizer.set_certifier_params(self.config.ac_params)
        # Run certification
        result = self.localizer.certify_solution(
            T_est=T_est.to("cpu").double().numpy(),
            verbose=self.config.verbose,
            adjust_cost=self.config.adjust_cost,
        )
        return result
