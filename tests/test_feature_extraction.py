import sys
from pathlib import Path
import pytest
import torch
import numpy as np
import os

# Ensure plotting uses the desired X display (useful in headless CI/devcontainer)
os.environ["DISPLAY"] = ":32"

from stereo_loc.EurocProcess import EurocDataset
from stereo_loc.FeatureExtractorAndMatcher import (
    FeatureExtractorConfig,
    FeatureMatcherConfig,
    FeatureExtractorAndMatcher,
)
from lightglue import viz2d

from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="module")
def rectified_image_tensors():
    """Instantiate a EurocDataset, retrieve a rectified stereo pair and convert to torch tensors."""

    # Expected default dataset location inside the experiments tree
    default_root = ROOT / "data" / "Euroc" / "MH_01_easy"
    if not default_root.exists():
        pytest.skip(f"Euroc dataset not found at {default_root}")

    ds = EurocDataset(default_root)
    # pick first available timestamp
    timestamps = list(ds.cam0.timestamp_to_file.keys())
    if not timestamps:
        pytest.skip("No images found in Euroc cam0 data mapping")

    timestamp = timestamps[0]
    img0, img1 = ds.get_image_at_timestamp(timestamp, rectify=True)

    # ensure single-channel float tensors shaped (B, C, H, W)
    def to_tensor(img: np.ndarray) -> torch.Tensor:
        if img.ndim == 3:
            # convert to grayscale by averaging channels
            img = img.mean(axis=2)
        img_t = torch.from_numpy(img.astype(np.float32))
        img_t = img_t.unsqueeze(0)  # (1,H,W)
        img_t = img_t / 255.0
        return img_t

    t0 = to_tensor(img0)
    t1 = to_tensor(img1)
    return t0, t1


def test_feature_extraction_and_matching_shapes(rectified_image_tensors, plot=True):
    # Just use right and left images from rectified stereo pair
    im0, im1 = rectified_image_tensors

    extractor_cfg = FeatureExtractorConfig(device="cuda", max_num_keypoints=256)
    matcher_cfg = FeatureMatcherConfig(device="cuda", match_threshold=0.2)
    model = FeatureExtractorAndMatcher(extractor_cfg, matcher_cfg)

    m0, m1 = model.forward(im0, im1)

    # Basic shape assertions
    assert isinstance(m0, torch.Tensor)
    assert isinstance(m1, torch.Tensor)
    assert m0.ndim == 2 and m1.ndim == 2  # (N, 2) for keypoint coordinates
    assert m0.shape[0] == m1.shape[0]  # same number of matches
    assert m0.shape[1] == 2 and m1.shape[1] == 2  # (x,y) per keypoint

    # Assert that the matched keypoints have the same y pixel coordinate (since images are rectified)
    y_diff = torch.abs(m0[:, 1] - m1[:, 1])
    assert torch.sum(
        y_diff > 1.0
    ) < 10, "Matched keypoints should have nearly identical y coordinates in rectified stereo images, with some outliers allowed"

    if plot:
        axes = viz2d.plot_images([im0, im1])
        viz2d.plot_matches(m0, m1, color="lime", lw=0.2)
        plt.show()
