import torch.nn as nn
from dataclasses import dataclass, field
from enum import Enum
import torch
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path


from stereo_loc.FeatureExtractorAndMatcher import (
    FeatureExtractorConfig,
    FeatureMatcherConfig,
    FeatureExtractorAndMatcher,
)
from utils.stereo_camera_model import StereoCameraModel, StereoCameraConfig
from stereo_loc.DataAssociationBlocks import (
    DataAssociationBlock,
    DataAssociationMethod,
    DataAssociationConfig,
)
from utils.keypoint_tools import get_inv_cov_weights
from stereo_loc.PointCloudRegistrationBlock import (
    PointCloudRegistrationBlock,
    PointCloudRegistrationConfig,
)
from ranktools import AnalyticCenterResult

torch.set_default_dtype(torch.float32)


@dataclass
class StereoPipelineConfig:
    """Configuration for the stereo pipeline."""

    # Top level Verbosity
    verbose: bool = True
    # Debug flag for the stereo pipeline. If true, will output additional debug information and visualizations.
    debug: bool = False

    # submodule configs
    feature_extractor_config: FeatureExtractorConfig = field(
        default_factory=FeatureExtractorConfig
    )
    feature_matcher_config: FeatureMatcherConfig = field(
        default_factory=FeatureMatcherConfig
    )
    stereo_camera_config: StereoCameraConfig = field(default_factory=StereoCameraConfig)
    data_association_config: DataAssociationConfig = field(
        default_factory=DataAssociationConfig
    )
    registration_config: PointCloudRegistrationConfig = field(
        default_factory=PointCloudRegistrationConfig
    )


def load_config(override_path: Path | None = None) -> StereoPipelineConfig:
    # Start with defaults from dataclass
    config = OmegaConf.structured(StereoPipelineConfig)

    # Merge overrides if provided
    if override_path:
        overrides = OmegaConf.load(override_path)
        config = OmegaConf.merge(config, overrides)

    return OmegaConf.to_object(config)


@dataclass
class StereoPipelineDebugInfo:
    """Debug information for the stereo pipeline."""

    # Matched keypoints in pixel coordinates, of shape (2, N, 2).
    keypoints_2D: torch.Tensor = None
    # 3D keypoints in the sensor frame (left camera frame) given in homogeneous coordinates, of shape (2, 4, N).
    keypoints_3D: torch.Tensor = None
    # Inliers from the data association step, of shape (N,).
    inliers: torch.Tensor = None
    # Inverse covariance weights for each matched point pair, of shape (N, 3, 3).
    inv_cov_weights: torch.Tensor = None
    # Certification results
    cert_result_association: AnalyticCenterResult | None = None
    cert_result_registration: AnalyticCenterResult | None = None
    # Data association solution (for CLIPPER_SDP), of shape (N, N).
    da_soln: np.ndarray | None = None
    # Clipper matrix M (for CLIPPER_SDP), of shape (N, N).
    M: np.ndarray | None = None


@dataclass
class StereoPipelineOutput:
    """Output of the stereo pipeline."""

    # Relative transform between the robot body frame and the camera frame, of shape (4, 4).
    relative_transform: torch.Tensor
    # Certification flags
    data_association_certified: bool = False
    registration_certified: bool = False
    # Inlier count
    num_inliers: int = 0
    # Additional information from the registration block, such as the number of inliers, etc.
    registration_info: dict = None
    # Debug information for the stereo pipeline.
    debug_info: StereoPipelineDebugInfo | None = None


class StereoPipeline:
    """Stereo pipeline that takes in rectified stereo images and disparities and produces the relative transform between the robot body frame.
    Note that this pipeline is inference only and cannot be used for training."""

    def __init__(self, config: StereoPipelineConfig):
        self.config = config

        # Set up feature extractor and matcher and move to GPU if available
        self.feature_extractor_and_matcher = FeatureExtractorAndMatcher(
            self.config.feature_extractor_config,
            self.config.feature_matcher_config,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Set up stereo camera model
        self.stereo_camera_model = StereoCameraModel(self.config.stereo_camera_config)

        # Set up data association
        self.data_association: DataAssociationBlock
        if self.config.data_association_config.method in [
            DataAssociationMethod.CLIPPER,
            DataAssociationMethod.CLIPPER_SDP,
        ]:
            self.data_association = DataAssociationBlock(
                self.config.data_association_config
            )
        elif self.config.data_association_config.method == DataAssociationMethod.RANSAC:
            raise NotImplementedError("RANSAC data association not implemented yet.")
        else:
            raise ValueError(
                f"Invalid data association method: {self.config.data_association_config.method}"
            )

    def forward(
        self, images: torch.Tensor, disparities: torch.Tensor, T_init: np.ndarray
    ) -> StereoPipelineOutput:
        """Forward pass through the stereo pipeline.
        Args:
            images (list of torch.Tensor): List of two rectified images corresponding to the differen poses, each of shape (C, H, W).
            disparities (list of torch.Tensor): List of two disparity maps corresponding to the stereo images, each of shape (H, W).
            T_init (torch.Tensor): Initial guess for the relative transform from the target to the source frames, T_src_trg, of shape (4, 4).
        Returns:
            relative_transform (torch.Tensor): Relative transform between the robot body frame and the camera frame, of shape (4, 4).
        """

        # Call feature extractor and matcher to get matched keypoints in pixel coordinates
        kpt_2D_src, kpt_2D_trg = self.feature_extractor_and_matcher.forward(
            images[0], images[1]
        )
        # Reshape keypoints to (1, 2, N) for the inverse camera model
        kpt_2D_src = kpt_2D_src.unsqueeze(0).transpose(1, 2)  # (1, 2, N)
        kpt_2D_trg = kpt_2D_trg.unsqueeze(0).transpose(1, 2)  # (1, 2, N)
        if self.config.verbose:
            print(f"Number of matched keypoints: {kpt_2D_src.size(2)}")

        # Get 3D keypoints from the disparities
        kpt_3D_src, valid_src = self.stereo_camera_model.inverse_camera_model(
            kpt_2D_src, disparities[0]
        )
        kpt_3D_trg, valid_trg = self.stereo_camera_model.inverse_camera_model(
            kpt_2D_trg, disparities[1]
        )
        kpt_3D_src = kpt_3D_src.squeeze(0)  # (4,N)
        kpt_3D_trg = kpt_3D_trg.squeeze(0)  # (4,N)

        # Restrict to valid keypoints
        valid_src = valid_src.squeeze(0).squeeze(0)  # (N,)
        valid_trg = valid_trg.squeeze(0).squeeze(0)  # (N,)
        valid = valid_src & valid_trg  # (N,)
        kpt_3D_src = kpt_3D_src[:, valid]  # (4, M)
        kpt_3D_trg = kpt_3D_trg[:, valid]  # (4, M)
        if self.config.verbose:
            print("Done generating 3D keypoints from disparities.")
            print(f"Number of valid 3D keypoints: {kpt_3D_src.size(1)}")

        # Call 3D data association to get inliers
        if self.config.data_association_config.method == DataAssociationMethod.CLIPPER:
            inliers, soln = self.data_association.run_clipper(kpt_3D_src, kpt_3D_trg)
        elif self.config.data_association_config.method == DataAssociationMethod.RANSAC:
            raise NotImplementedError("RANSAC data association not implemented yet.")
        elif (
            self.config.data_association_config.method
            == DataAssociationMethod.CLIPPER_SDP
        ):
            inliers, soln = self.data_association.run_sdp(kpt_3D_src, kpt_3D_trg)
        else:
            raise ValueError(
                f"Invalid data association method: {self.config.data_association_config.method}"
            )

        if self.config.verbose:
            print(f"Number of inliers after data association: {torch.sum(inliers)}")
        # Call 3D data association certifier module
        data_association_certified = False
        cert_result_da = None
        if self.config.data_association_config.certify:
            # if thresholding (converting to max clique problem), pass the inliers.
            if self.config.data_association_config.unweighted:
                cert_kwargs = dict(inliers=inliers)
            else:
                cert_kwargs = dict(soln=soln)
            cert_result_da = self.data_association.certify_solution(**cert_kwargs)
            data_association_certified = cert_result_da.certified
        if self.config.verbose:
            print(f"Data association certification result: {cert_result_da}")
        # Restrict points to inliers
        kpt_3D_src_inlier = kpt_3D_src[:, inliers]  # (4, K)
        kpt_3D_trg_inlier = kpt_3D_trg[:, inliers]  # (4, K)
        # Retrieve matrix weights for each matched point pair (expects batch dim)
        valid_dummy = torch.ones(
            1, 1, kpt_3D_src_inlier.size(1), device=kpt_3D_src_inlier.device, dtype=bool
        )
        inv_cov_weights, cov_cam = get_inv_cov_weights(
            kpt_3D_src_inlier.unsqueeze(0),
            valid_dummy,
            self.stereo_camera_model,
            normalize_weights=True,
        )

        if self.config.verbose:
            print(f"Starting registration with {kpt_3D_src_inlier.size(1)} inliers.")
        # Call pose estimator to get the relative transform between the robot body frame and the camera frame
        registration_block = PointCloudRegistrationBlock(
            config=self.config.registration_config,
            keypoints_3D_src=kpt_3D_src_inlier[:3, :],  # (3, K)
            keypoints_3D_trg=kpt_3D_trg_inlier[:3, :],  # (3, K)
            inv_cov_weights=inv_cov_weights.squeeze(0),  # (K, 3, 3)
        )
        T_est, info = registration_block.solve_factor_graph(
            T_init, verbose=self.config.debug
        )

        # Certify solution
        registration_certified = False
        cert_result_reg = None
        if self.config.registration_config.certify:
            cert_result_reg = registration_block.certify_solution(T_est)
            if self.config.verbose:
                print(f"Certification result: {cert_result_reg}")
            registration_certified = cert_result_reg.certified

        # Generate standard output
        output = StereoPipelineOutput(
            relative_transform=T_est,  # (4, 4)
            data_association_certified=data_association_certified,
            registration_certified=registration_certified,
            registration_info=info,
            num_inliers=torch.sum(inliers).item(),
        )

        # Generate debug output if requested
        if self.config.debug:
            debug_info = StereoPipelineDebugInfo(
                keypoints_2D=torch.stack(
                    [kpt_2D_src.squeeze(0), kpt_2D_trg.squeeze(0)], dim=0
                ),  # (2, N, 2)
                keypoints_3D=torch.stack([kpt_3D_src, kpt_3D_trg], dim=0),  # (2, 4, N)
                inliers=inliers,  # (N,)
                inv_cov_weights=inv_cov_weights.squeeze(0),  # (K, 3, 3)
                cert_result_association=cert_result_da,
                cert_result_registration=cert_result_reg,
                da_soln=soln,
                M=self.data_association.M,
            )
            output.debug_info = debug_info

        return output


if __name__ == "__main__":
    # Test load config function
    ROOT = Path(__file__).resolve().parents[2]
    config_file = ROOT / "configs" / "euroc_certify_reg.yaml"
    config = load_config(config_file)
    print(config)
