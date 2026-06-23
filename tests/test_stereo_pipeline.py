import sys
from pathlib import Path
import pytest
import torch
import numpy as np
import os
from pylgmath import Transformation

# Ensure plotting uses the desired X display (useful in headless CI/devcontainer)
os.environ["DISPLAY"] = ":32"

from stereo_loc.EurocProcess import EurocDataset
from stereo_loc.FeatureExtractorAndMatcher import (
    FeatureExtractorConfig,
    FeatureMatcherConfig,
    FeatureExtractorAndMatcher,
)
from stereo_loc.StereoPipeline import StereoPipeline, StereoPipelineConfig
from stereo_loc.DataAssociationBlocks import ClipperBlock, ClipperConfig
from utils.stereo_camera_model import (
    StereoCameraConfig,
    StereoCameraModel,
    get_disparity,
)
from stereo_loc.PointCloudRegistrationBlock import (
    PointCloudRegistrationBlock,
    PointCloudRegistrationConfig,
)
from utils.keypoint_tools import get_inv_cov_weights

from lightglue import viz2d
from open3d.io import read_point_cloud
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="module")
def euroc_data():
    """Instantiate a EurocDataset, retrieve a rectified stereo pair and convert to torch tensors."""

    # Expected default dataset location inside the experiments tree
    default_root = ROOT / "data" / "Euroc" / "MH_01_easy"
    if not default_root.exists():
        pytest.skip(f"Euroc dataset not found at {default_root}")

    ds = EurocDataset(default_root)
    return ds


def get_euroc_stereo_image(ds, timestamp):
    # pick first available timestamp
    img0, img1 = ds.get_image_at_timestamp(timestamp, rectify=True)
    t0 = image_to_tensor(img0)
    t1 = image_to_tensor(img1)
    return t0, t1, img0, img1


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    # Convert ndarray image to torch tensor, ensuring it's single-channel and normalized to [0,1]
    if img.ndim == 3:
        # convert to grayscale by averaging channels
        img = img.mean(axis=2)
    img_t = torch.from_numpy(img.astype(np.float32))
    img_t = img_t.unsqueeze(0)  # (1,H,W)
    img_t = img_t / 255.0
    return img_t


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
def bunny_stereo_synthetic(plot: bool = False):
    """Load points from bun10k.ply, create two z-offset cameras, and project stereo image coords."""

    pcfile = ROOT / "data" / "bun10k.ply"
    if not pcfile.exists():
        pytest.skip(f"Point cloud not found at {pcfile}")

    pcd = read_point_cloud(str(pcfile))
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        pytest.skip(f"No points found in {pcfile}")

    num_points = min(100, points.shape[0])
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(points.shape[0], size=num_points, replace=False)
    sampled_points = points[sample_idx]

    n_outliers = 2

    centroid = sampled_points.mean(axis=0)
    z_extent = float(np.max(sampled_points[:, 2]) - np.min(sampled_points[:, 2]))
    z_offset = max(2.0 * z_extent, 1.0)
    baseline = max(0.1 * z_extent, 0.05)

    # Two camera frames, both looking along +z at the sampled points
    c1 = centroid + np.array([0.0, 0.0, -z_offset], dtype=np.float32)
    c2 = centroid + np.array([0.0, 0.0, -(z_offset + baseline)], dtype=np.float32)

    T_w_c1 = _make_camera_transform(c1)
    T_w_c2 = _make_camera_transform(c2)
    T_21 = np.linalg.inv(T_w_c2) @ T_w_c1

    outlier_scale = max(z_extent, 1.0)
    outliers_1 = centroid + rng.uniform(
        low=-4.0 * outlier_scale,
        high=4.0 * outlier_scale,
        size=(n_outliers, 3),
    )
    outliers_2 = centroid + rng.uniform(
        low=-4.0 * outlier_scale,
        high=4.0 * outlier_scale,
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

    if plot:
        # Plot the sampled 3D points from the perspective of each camera frame.
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection="3d")
        ax1.scatter(
            points_c1[:, 0],
            points_c1[:, 1],
            points_c1[:, 2],
            c="blue",
            s=10,
            label="frame 1",
        )
        ax1.set_title("Bunny points in camera frame 1")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.legend()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.scatter(
            points_c2[:, 0],
            points_c2[:, 1],
            points_c2[:, 2],
            c="red",
            s=10,
            label="frame 2",
        )
        ax2.set_title("Bunny points in camera frame 2")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.legend()

        # Plot stereo views as subplots in a single figure.
        fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
        stereo_views = [
            (stereo_img_coords_1, "frame 1"),
            (stereo_img_coords_2, "frame 2"),
        ]
        all_u = np.concatenate(
            [
                stereo_img_coords_1[[0, 2], :].ravel(),
                stereo_img_coords_2[[0, 2], :].ravel(),
            ]
        )
        all_v = np.concatenate(
            [
                stereo_img_coords_1[[1, 3], :].ravel(),
                stereo_img_coords_2[[1, 3], :].ravel(),
            ]
        )
        u_min, u_max = float(np.min(all_u)), float(np.max(all_u))
        v_min, v_max = float(np.min(all_v)), float(np.max(all_v))
        for ax, (stereo_coords, title) in zip(axes, stereo_views):
            ax.scatter(
                stereo_coords[0, :],
                stereo_coords[1, :],
                c="blue",
                s=10,
                label="left",
            )
            ax.scatter(
                stereo_coords[2, :],
                stereo_coords[3, :],
                c="red",
                s=10,
                label="right",
            )
            ax.set_xlim(u_min, u_max)
            ax.set_ylim(v_max, v_min)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Stereo image coordinates for {title}")
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            ax.legend()

        fig3.tight_layout()
        plt.show()

    return {
        "T_21": T_21,
        "n_outliers": n_outliers,
        "stereo_image_coords": {
            "frame_1": stereo_img_coords_1,
            "frame_2": stereo_img_coords_2,
        },
        "stereo_model": stereo_model,
        "points_1": points_c1,
        "points_2": points_c2,
    }


def test_feature_extraction_and_matching_euroc(euroc_data, plot=False):
    # Just use right and left images from rectified stereo pair
    ds = euroc_data
    timestamp = ds.cam0.timestamps[1000]
    im0, im1, _, _ = get_euroc_stereo_image(ds, timestamp)

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
    assert (
        torch.sum(y_diff > 1.0) < 10
    ), "Matched keypoints should have nearly identical y coordinates in rectified stereo images, with some outliers allowed"

    if plot:
        axes = viz2d.plot_images([im0, im1])
        viz2d.plot_matches(m0, m1, color="lime", lw=0.2)
        plt.show()


def test_3d_point_reconstruction_euroc(euroc_data, plot=True):
    # dataset
    ds = euroc_data
    # indices for test
    index0 = 1000
    index1 = 1010
    # Get timestamps from the camera data at given indices
    timestamp0 = ds.cam0.timestamps[index0]
    timestamp1 = ds.cam1.timestamps[index1]
    # Get ground truth relative transform between cameras
    T_01 = ds.get_relative_transform(timestamp0, timestamp1, camera_frame=True)
    T_01 = torch.Tensor(T_01).float().to("cuda")  # (4,4)
    # Get rectified stereo images
    im0_L_t, im0_R_t, im0_L, im0_R = get_euroc_stereo_image(euroc_data, timestamp0)
    im1_L_t, im1_R_t, im1_L, im1_R = get_euroc_stereo_image(euroc_data, timestamp1)
    # Get disparity maps
    disp0 = get_disparity(im0_L, im0_R).to("cuda")
    disp1 = get_disparity(im1_L, im1_R).to("cuda")

    # Get feature locations
    extractor_cfg = FeatureExtractorConfig(device="cuda", max_num_keypoints=256)
    matcher_cfg = FeatureMatcherConfig(device="cuda", match_threshold=0.2)
    model = FeatureExtractorAndMatcher(extractor_cfg, matcher_cfg)
    m0, m1 = model.forward(im0_L_t, im1_L_t)
    m0 = m0.unsqueeze(0).transpose(1, 2)  # (1,2,N)
    m1 = m1.unsqueeze(0).transpose(1, 2)  # (1,2,N)
    # set up stereo model
    stereo_camera_params = StereoCameraConfig(
        cu=ds.stereo_camera.camera_cx,
        cv=ds.stereo_camera.camera_cy,
        f=ds.stereo_camera.camera_fx,
        b=ds.stereo_camera.camera_bf / ds.stereo_camera.camera_fx,  # baseline in meters
        sigma=0.5,  # disparity noise in pixels
    )
    stereo_camera = StereoCameraModel(stereo_camera_params)
    # Get 3D keypoints from the disparities
    kpt_3D_0, valid_0 = stereo_camera.inverse_camera_model(m0, disp0)
    kpt_3D_1, valid_1 = stereo_camera.inverse_camera_model(m1, disp1)
    kpt_3D_0 = kpt_3D_0.squeeze(0)  # (4,N)
    kpt_3D_1 = kpt_3D_1.squeeze(0)  # (4,N)
    # Get only valid keypoints (those with valid disparities in both frames)
    valid_0 = valid_0.squeeze(0).squeeze(0)  # (N,)
    valid_1 = valid_1.squeeze(0).squeeze(0)  # (N,)
    valid_matches = valid_0 & valid_1
    kpt_3D_0 = kpt_3D_0[:, valid_matches]
    kpt_3D_1 = kpt_3D_1[:, valid_matches]

    # Map 3D keypoints from frame 1 to frame 0 using the ground truth transform
    kpt_3D_1_in_0 = T_01 @ kpt_3D_1

    # Verify that there are some inliers
    kpt_diff = kpt_3D_0 - kpt_3D_1_in_0
    tolerance = 10e-2  # 10 cm error tolerance
    num_inliers = torch.sum(torch.norm(kpt_diff[:3, :], dim=0) < tolerance).item()
    assert (
        num_inliers > 10
    ), f"Expected at least 10 inliers within {tolerance}m, but got {num_inliers}"

    # Run Clipper block to test data association on the reconstructed 3D points. We expect most of the valid matches to be inliers, and the outliers to be rejected.
    clipper_cfg = ClipperConfig()
    clipper_cfg.invariant_epsilon = (
        0.2  # set epsilon to 20 cm to account for noise in the reconstructed 3D points
    )
    clipper_cfg.invariant_sigma = (
        0.1  # set sigma to 10 cm to allow for some noise in the pairwise distances
    )
    clipper_block = ClipperBlock(clipper_cfg)
    inliers = clipper_block.forward(kpt_3D_0, kpt_3D_1)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        inlier_mask = inliers.bool()
        outlier_mask = ~inlier_mask
        ax.scatter(
            kpt_3D_0[0, :].cpu(),
            kpt_3D_0[1, :].cpu(),
            kpt_3D_0[2, :].cpu(),
            c="blue",
            s=10,
            label="frame 0",
        )

        ax.scatter(
            kpt_3D_1_in_0[0, inlier_mask].cpu(),
            kpt_3D_1_in_0[1, inlier_mask].cpu(),
            kpt_3D_1_in_0[2, inlier_mask].cpu(),
            c="lime",
            s=10,
            label="inliers (frame 1 transformed)",
        )
        ax.scatter(
            kpt_3D_1_in_0[0, outlier_mask].cpu(),
            kpt_3D_1_in_0[1, outlier_mask].cpu(),
            kpt_3D_1_in_0[2, outlier_mask].cpu(),
            c="red",
            s=10,
            label="outliers (frame 1 transformed)",
        )
        # Draw green lines for inliers and red lines for outliers.
        for i in range(kpt_3D_0.shape[1]):
            line_color = "lime" if bool(inliers[i]) else "red"
            ax.plot(
                [kpt_3D_0[0, i].cpu(), kpt_3D_1_in_0[0, i].cpu()],
                [kpt_3D_0[1, i].cpu(), kpt_3D_1_in_0[1, i].cpu()],
                [kpt_3D_0[2, i].cpu(), kpt_3D_1_in_0[2, i].cpu()],
                c=line_color,
                linewidth=0.5,
                alpha=0.5,
            )

        ax.set_title("Reconstructed 3D keypoints from stereo")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        plt.show()


def test_clipper_block(bunny_stereo_synthetic):
    # T_21 = bunny_stereo_synthetic["T_21"]
    # stereo_img_coords_1 = bunny_stereo_synthetic["stereo_image_coords"]["frame_1"]
    # stereo_img_coords_2 = bunny_stereo_synthetic["stereo_image_coords"]["frame_2"]
    # stereo_model = bunny_stereo_synthetic["stereo_image_coords"]["stereo_model"]
    points_c1 = bunny_stereo_synthetic["points_1"]
    points_c2 = bunny_stereo_synthetic["points_2"]
    n_outliers = bunny_stereo_synthetic["n_outliers"]

    # Instantiate CLIPPER block
    clipper_cfg = ClipperConfig()
    clipper_block = ClipperBlock(clipper_cfg)
    inliers = clipper_block.forward(
        torch.from_numpy(points_c1.T).float().to("cpu"),  # (4,N)
        torch.from_numpy(points_c2.T).float().to("cpu"),  # (4,N)
    )
    # assert that the original sampled points are inliers and the synthetic outliers are rejected
    assert (
        torch.sum(inliers) >= points_c1.shape[0] - n_outliers
    ), f"Expected at least {points_c1.shape[0] - n_outliers} inliers, but got {torch.sum(inliers)} inliers out of {points_c1.shape[0]} total points"


def test_pointcloud_registration(bunny_stereo_synthetic):
    points_c1 = bunny_stereo_synthetic["points_1"]
    points_c2 = bunny_stereo_synthetic["points_2"]
    n_outliers = bunny_stereo_synthetic["n_outliers"]

    # remove outliers from points_c1 and points_c2 for registration
    points_c1_inliers = torch.Tensor(points_c1[:-n_outliers, :].T).float()
    points_c2_inliers = torch.Tensor(points_c2[:-n_outliers, :].T).float()

    # Generate matrix weights
    stereo_cam = bunny_stereo_synthetic["stereo_model"]
    inv_cov_weights, cov_cam = get_inv_cov_weights(
        points_c1_inliers.unsqueeze(0),
        torch.ones((1, 1, points_c1_inliers.shape[1]), dtype=torch.bool),
        stereo_cam,
        normalize_weights=True,
    )
    # Run PointCloudRegistrationBlock to estimate the relative transform
    config = PointCloudRegistrationConfig()
    pcr = PointCloudRegistrationBlock(
        config, points_c1_inliers, points_c2_inliers, inv_cov_weights.squeeze(0)
    )
    T_est, info = pcr.solve_factor_graph(
        torch.eye(4, dtype=torch.float32), verbose=True
    )
    # Check that we are close to the ground truth transform
    T_gt = bunny_stereo_synthetic["T_21"]
    T_error = np.linalg.norm(T_est - T_gt)
    assert (
        T_error < 0.1
    ), f"Estimated transform is too far from ground truth: error={T_error:.4f} (should be < 0.1)"


def test_stereo_pipeline_no_cert(euroc_data):
    """Test the full stereo localization pipeline on a pair of Euroc images.
    No Certification performed."""
    with torch.no_grad():
        ds = euroc_data
        # Compute transform for one step
        timestamp0 = ds.cam0.timestamps[1000]
        timestamp1 = ds.cam1.timestamps[1001]
        T_01 = ds.get_relative_transform(timestamp0, timestamp1, camera_frame=True)
        im0_L_t, im0_R_t, im0_L, im0_R = get_euroc_stereo_image(ds, timestamp0)
        im1_L_t, im1_R_t, im1_L, im1_R = get_euroc_stereo_image(ds, timestamp1)
        images = [im0_L_t.float(), im1_L_t.float()]
        disp0 = get_disparity(im0_L, im0_R).float().to("cuda")
        disp1 = get_disparity(im1_L, im1_R).float().to("cuda")

        # Set up config
        config = StereoPipelineConfig()
        # Get stereo camera config from dataset
        config.stereo_camera_config = ds.get_stereo_cam_config()
        # Set up the stereo pipeline
        pipeline = StereoPipeline(config)
        # Initialize with ground truth transform
        T_src_trg_gt = Transformation(T_ba=T_01)
        output = pipeline.forward(
            images=images,
            disparities=[disp0.float(), disp1.float()],
            T_init=T_src_trg_gt.matrix(),  # initial guess for the relative transform
        )

        # Check that the estimated transform is close to the ground truth
        T_src_trg = Transformation(T_ba=output.relative_transform)
        xi_error = (T_src_trg.inverse() @ T_src_trg_gt).vec()
        trans_error = np.linalg.norm(xi_error[:3])
        rot_error = np.linalg.norm(xi_error[3:])
        assert (
            trans_error < 0.1
        ), f"Estimated translation error is too large: {trans_error:.4f} (should be < 0.1)"
        assert (
            rot_error < 0.01
        ), f"Estimated rotation error is too large: {rot_error:.4f} (should be < 0.1)"
