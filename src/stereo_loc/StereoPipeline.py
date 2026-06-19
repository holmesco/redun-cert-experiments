import torch.nn as nn
from dataclasses import dataclass

from FeatureExtractorAndMatcher import FeatureExtractorConfig, FeatureMatcherConfig, FeatureExtractorAndMatcher 
from utils.stereo_camera_model import StereoCameraModel

@dataclass
class StereoPipelineConfig:
    """ Configuration for the stereo pipeline."""
    feature_extractor_config: FeatureExtractorConfig
    feature_matcher_config: FeatureMatcherConfig
    
    


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
        
    def forward(self, images, disparities):
        
        # Call feature extractor and matcher to get matched keypoints in pixel coordinates
        kpt_2D_src, kpt_2D_trg = self.feature_extractor_and_matcher.forward(images[0], images[1])
        
        # TODO: Get 3D keypoints from the disparities
        # Make sure to mask points that have invalid disparities
        
        # TODO: Call 3D data association using CLIPPER
        
        # TODO: Call 3D data association certifier module
        
        # TODO: Retrieve matrix weights for each matched point pair
        
        # TODO: Call pose estimator to get the relative transform between the robot body frame and the camera frame