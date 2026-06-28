import sys
from pathlib import Path
import pytest
import numpy as np
import torch

from open3d.io import read_point_cloud
from utils.stereo_camera_model import StereoCameraConfig, StereoCameraModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _make_camera_transform(position: np.ndarray) -> np.ndarray:
    """Create a camera-to-world transform with identity rotation and the given position."""
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = position.astype(np.float32)
    return T


def _world_to_camera(points_w: np.ndarray, T_w_c: np.ndarray) -> np.ndarray:
    """Transform world points into the camera frame."""
    T_c_w = np.linalg.inv(T_w_c)
    points_h = np.hstack(
        [points_w.astype(np.float32), np.ones((points_w.shape[0], 1), dtype=np.float32)]
    )
    points_c = (T_c_w @ points_h.T).T
    return points_c


def _project_stereo_points(
    model: StereoCameraModel, points_c: np.ndarray
) -> torch.Tensor:
    """Project camera-frame points to stereo image coordinates using StereoCameraModel."""
    cam_coords = torch.from_numpy(points_c.T).unsqueeze(0).float()  # (1,4,N)
    M = model.M.unsqueeze(0)  # (1,4,4)
    img_coords = model.camera_to_image(cam_coords, M)  # (1,4,N)
    return img_coords.squeeze(0).cpu()  # (4,N)


@pytest.fixture(scope="module")
def bunny_stereo_synthetic():
    """Load points from bun10k.ply, create two z-offset cameras, and project stereo image coords."""

    pcfile = ROOT / "data" / "bun10k.ply"
    if not pcfile.exists():
        pytest.skip(f"Point cloud not found at {pcfile}")

    pcd = read_point_cloud(str(pcfile))
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        pytest.skip(f"No points found in {pcfile}")

    num_points = min(50, points.shape[0])
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(points.shape[0], size=num_points, replace=False)
    sampled_points = points[sample_idx]

    n_outliers = 20

    centroid = sampled_points.mean(axis=0)
    z_extent = float(np.max(sampled_points[:, 2]) - np.min(sampled_points[:, 2]))
    z_offset = max(2.0 * z_extent, 1.0)
    baseline = max(0.1 * z_extent, 0.05)

    # Two camera frames, both looking along +z at the sampled points
    c1 = centroid + np.array([0.0, 0.0, -z_offset], dtype=np.float32)
    c2 = centroid + np.array([0.0, 0.0, -(z_offset + baseline)], dtype=np.float32)

    T_w_c1 = _make_camera_transform(c1)
    T_w_c2 = _make_camera_transform(c2)
    T_12 = np.linalg.inv(T_w_c1) @ T_w_c2

    outlier_scale = 10.0
    centroid_1 = centroid + np.array([0.0, 1.0, 0.0], dtype=np.float32)
    outliers_1 = centroid_1 + rng.uniform(
        low=-outlier_scale,
        high=outlier_scale,
        size=(n_outliers, 3),
    )
    centroid_2 = centroid + np.array([0.0, -0.5, 0.0], dtype=np.float32)
    outliers_2 = centroid_2 + rng.uniform(
        low=-outlier_scale,
        high=outlier_scale,
        size=(n_outliers, 3),
    )

    points_1 = np.concatenate([sampled_points, outliers_1], axis=0)
    points_2 = np.concatenate([sampled_points, outliers_2], axis=0)

    points_c1 = _world_to_camera(points_1, T_w_c1)
    points_c2 = _world_to_camera(points_2, T_w_c2)

    stereo_cfg = StereoCameraConfig(
        cu=320.0,
        cv=240.0,
        f=525.0,
        b=0.1,
        sigma=0.5,
    )
    stereo_model = StereoCameraModel(stereo_cfg)

    stereo_img_coords_1 = _project_stereo_points(stereo_model, points_c1)
    stereo_img_coords_2 = _project_stereo_points(stereo_model, points_c2)

    return {
        "T_12": T_12,
        "n_outliers": n_outliers,
        "stereo_image_coords": {
            "frame_1": stereo_img_coords_1,
            "frame_2": stereo_img_coords_2,
        },
        "stereo_model": stereo_model,
        "points_1": points_c1,
        "points_2": points_c2,
    }
