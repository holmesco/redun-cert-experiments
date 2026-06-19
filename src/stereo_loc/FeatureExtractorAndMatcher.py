from dataclasses import dataclass

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
import torch


@dataclass
class FeatureExtractorConfig:
    """Configuration for the feature extractor."""

    model_name: str = "superpoint"
    device: str = "cuda"  # or "cpu"
    max_num_keypoints: int = 1024  # Maximum number of keypoints to extract


@dataclass
class FeatureMatcherConfig:
    """Configuration for the feature matcher."""

    model_name: str = "lightglue"
    device: str = "cuda"  # or "cpu"
    match_threshold: float = 0.2  # Threshold for matching keypoints


class FeatureExtractorAndMatcher(torch.nn.Module):
    """Feature extractor and matcher that takes in rectified stereo images and produces matched keypoints.
    This class is a wrapper around the SuperPoint feature extractor and LightGlue feature matcher.
    """

    def __init__(
        self,
        feature_extractor_config: FeatureExtractorConfig,
        feature_matcher_config: FeatureMatcherConfig,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor_config = feature_extractor_config
        self.feature_matcher_config = feature_matcher_config

        # Initialize the feature extractor
        if self.feature_extractor_config.model_name == "superpoint":
            self.feature_extractor = (
                SuperPoint(
                    max_num_keypoints=self.feature_extractor_config.max_num_keypoints,
                    device=self.feature_extractor_config.device,
                )
                .eval()
                .to(self.device)
            )
        else:
            raise ValueError(
                f"Unsupported feature extractor model: {self.feature_extractor_config.model_name}"
            )

        # Initialize the feature matcher
        if self.feature_matcher_config.model_name == "lightglue":
            self.feature_matcher = (
                LightGlue(
                    match_threshold=self.feature_matcher_config.match_threshold,
                    device=self.feature_matcher_config.device,
                )
                .eval()
                .to(self.device)
            )
        else:
            raise ValueError(
                f"Unsupported feature matcher model: {self.feature_matcher_config.model_name}"
            )

    def forward(self, images0: torch.Tensor, images1: torch.Tensor):
        """
        Forward pass through the feature extractor and matcher.
        Note: Images assumed to have no batch dimension

        Args:
            images0 (torch.Tensor): Input images of shape (C, H, W).
            images1 (torch.Tensor): Input images of shape (C, H, W).

        Returns:
            m_kpts0 (torch.Tensor): Matched keypoints from the first image of shape (N, 2).
            m_kpts1 (torch.Tensor): Matched keypoints from the second image of shape (N, 2).
        """
        # Extract keypoints and descriptors from both images
        feats0 = self.feature_extractor.extract(images0.to(self.device))
        feats1 = self.feature_extractor.extract(images1.to(self.device))
        # Extract Matches
        matches01 = self.feature_matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension
        # Retreive the matched keypoints and their corresponding matches
        kpts0, kpts1, matches = (
            feats0["keypoints"],
            feats1["keypoints"],
            matches01["matches"],
        )
        # Get matched keypoints
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return m_kpts0, m_kpts1
