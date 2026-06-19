from pathlib import Path
import sys, os

os.environ.setdefault("DISPLAY", ":32")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import torch
import numpy as np
from matplotlib import pyplot as plt

from stereo_loc.EurocProcess import EurocDataset
from stereo_loc.FeatureExtractorAndMatcher import (
    FeatureExtractorConfig,
    FeatureMatcherConfig,
    FeatureExtractorAndMatcher,
)
from lightglue import viz2d

# dataset
default_root = ROOT / "data" / "Euroc" / "MH_01_easy"
if not default_root.exists():
    raise SystemExit(f"Euroc dataset not found at {default_root}")

ds = EurocDataset(default_root)
timestamps = list(ds.cam0.timestamp_to_file.keys())
if not timestamps:
    raise SystemExit("No images found in Euroc cam0 data mapping")

img0, _ = ds.get_image_at_timestamp(timestamps[1000], rectify=True)
img1, _ = ds.get_image_at_timestamp(timestamps[1005], rectify=True)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 3:
        img = img.mean(axis=2)
    t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0) / 255.0  # (1,H,W)
    return t


im0 = to_tensor(img0)
im1 = to_tensor(img1)

extractor_cfg = FeatureExtractorConfig(device="cpu", max_num_keypoints=256)
matcher_cfg = FeatureMatcherConfig(device="cpu", match_threshold=0.1)
model = FeatureExtractorAndMatcher(extractor_cfg, matcher_cfg)

m0, m1 = model.forward(im0, im1)
print("matches:", m0.shape, m1.shape)

axes = viz2d.plot_images([im0, im1])
viz2d.plot_matches(m0, m1, color="lime", lw=0.2)
plt.show()
