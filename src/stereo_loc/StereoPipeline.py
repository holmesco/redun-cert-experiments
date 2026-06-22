import torch.nn as nn
from dataclasses import dataclass
from enum import Enum

from stereo_loc.FeatureExtractorAndMatcher import FeatureExtractorConfig, FeatureMatcherConfig, FeatureExtractorAndMatcher 
from utils.stereo_camera_model import StereoCameraModel, StereoCameraConfig
from stereo_loc.ClipperBlock import ClipperBlock, ClipperConfig 

def DataAssociationMethod(Enum):
    CLIPPER = "clipper"
    RANSAC = "ransac"

@dataclass
class StereoPipelineConfig:
    """ Configuration for the stereo pipeline."""
    
    data_association_method: DataAssociationMethod = DataAssociationMethod.CLIPPER
    
    # submodule configs
    feature_extractor_config: FeatureExtractorConfig = FeatureExtractorConfig()
    feature_matcher_config: FeatureMatcherConfig = FeatureMatcherConfig()
    stereo_camera_config: StereoCameraConfig = StereoCameraConfig()
    clipper_config: ClipperConfig = ClipperConfig()
    


class StereoPipeline(nn.Module):
    """ Stereo pipeline that takes in rectified stereo images and disparities and produces the relative transform between the robot body frame."""
    def __init__(self, config: StereoPipelineConfig):
        self.config = config
        
        # Set up feature extractor and matcher
        self.feature_extractor_and_matcher = FeatureExtractorAndMatcher(
            self.config.feature_extractor_config,
            self.config.feature_matcher_config,
        )

        # Set up stereo camera model
        self.stereo_camera_model = StereoCameraModel(self.config.stereo_camera_config)
        
        # Set up data association
        if self.config.data_association_method == DataAssociationMethod.CLIPPER:
            self.data_association_module = ClipperBlock(self.config.clipper_config)
        elif self.config.data_association_method == DataAssociationMethod.RANSAC:
            raise NotImplementedError("RANSAC data association not implemented yet.")
        else:
            raise ValueError(f"Invalid data association method: {self.config.data_association_method}")
        
    def forward(self, images, disparities):
        """ Forward pass through the stereo pipeline.
        Args:
            images (list of torch.Tensor): List of two rectified stereo images, each of shape (C, H, W).
            disparities (list of torch.Tensor): List of two disparity maps corresponding to the stereo images, each of shape (H, W).
        Returns:
            relative_transform (torch.Tensor): Relative transform between the robot body frame and the camera frame, of shape (4, 4).
        """
        
        # Call feature extractor and matcher to get matched keypoints in pixel coordinates
        kpt_2D_src, kpt_2D_trg = self.feature_extractor_and_matcher.forward(images[0], images[1])
        
        # Get 3D keypoints from the disparities
        kpt_3D_src, valid_src = self.stereo_camera_model.inverse_camera_model(kpt_2D_src, disparities[0])
        kpt_3D_trg, valid_trg = self.stereo_camera_model.inverse_camera_model(kpt_2D_trg, disparities[1])
        kpt_3D_src = kpt_3D_src.squeeze(0)  # (4,N)
        kpt_3D_trg = kpt_3D_trg.squeeze(0)  # (4,N)
        valid_src = valid_src.squeeze(0).squeeze(0)  # (N,)
        valid_trg = valid_trg.squeeze(0).squeeze(0)  # (N,)
        valid = valid_src & valid_trg  # (N,)
        # TODO: Call 3D data association using CLIPPER
        
        # TODO: Call 3D data association certifier module
        
        # TODO: Retrieve matrix weights for each matched point pair
        
        # TODO: Call pose estimator to get the relative transform between the robot body frame and the camera frame