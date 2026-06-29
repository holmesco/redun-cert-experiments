"""
This script runs the stereo pipeline on a pair of images from the Euroc dataset and visualizes the results.
It retrieves a rectified stereo pair, computes disparities, and runs the stereo pipeline to estimate the relative transform between the two frames. It also provides options to visualize the 2D and 3D matches, as well as the estimated and ground truth transforms.
"""
import sys
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader 
import numpy as np
import os
from pylgmath import Transformation

# Ensure plotting uses the desired X display (useful in headless CI/devcontainer)
os.environ["DISPLAY"] = ":32"

from stereo_loc.EurocPreprocess import EurocPreprocess
from stereo_loc.DataAssociationBlocks import DataAssociationMethod
from stereo_loc.StereoPipeline import StereoPipeline, load_config
from stereo_loc.EurocDataloader import EurocDataset

from lightglue import viz2d
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


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


def plot_pointclouds(
    kpt_3D_0, kpt_3D_2, inliers, ax=None, title="3D Keypoints", plot_outliers=False
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    outlier_mask = ~inliers
    ax.scatter(
        kpt_3D_0[0, inliers],
        kpt_3D_0[1, inliers],
        kpt_3D_0[2, inliers],
        c="blue",
        s=3,
        label="frame 0",
        alpha=0.7,
    )

    ax.scatter(
        kpt_3D_2[0, inliers],
        kpt_3D_2[1, inliers],
        kpt_3D_2[2, inliers],
        c="red",
        s=3,
        alpha=0.7,
        label="inliers (frame 1)",
    )
    if plot_outliers:
        ax.scatter(
            kpt_3D_0[0, outlier_mask],
            kpt_3D_0[1, outlier_mask],
            kpt_3D_0[2, outlier_mask],
            c="red",
            s=3,
            alpha=0.7,
            label="outliers (frame 1)",
        )
        ax.scatter(
            kpt_3D_2[0, outlier_mask],
            kpt_3D_2[1, outlier_mask],
            kpt_3D_2[2, outlier_mask],
            c="red",
            s=3,
            alpha=0.7,
            label="outliers (frame 1)",
        )
    # Draw lines between matches.
    for i in range(kpt_3D_0.shape[1]):
        line_color = "lime" if inliers[i] else "red"
        linewidth = 1.0 if inliers[i] else 0.5
        if inliers[i] or plot_outliers:
            ax.plot(
                [kpt_3D_0[0, i], kpt_3D_2[0, i]],
                [kpt_3D_0[1, i], kpt_3D_2[1, i]],
                [kpt_3D_0[2, i], kpt_3D_2[2, i]],
                c=line_color,
                linewidth=linewidth,
                alpha=0.5,
            )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    return ax


def run_stereo_pipeline(index=1000, frame_skip=10, plot=False):
    with torch.no_grad():
        print("Setting up dataset...")
        # Expected default dataset location inside the experiments tree
        default_root = ROOT / "data" / "Euroc" / "MH_01_easy"
        if not default_root.exists():
            raise FileNotFoundError(f"Euroc dataset not found at {default_root}")

        # get preprocessing object
        euroc_preproc = EurocPreprocess(default_root)
        # get dataset object
        euroc_dataset = EurocDataset(euroc_preproc, frame_interval=10)
        # Get the timestamps for the two frames       
        time, time_interval, img0_L, img1_L, disp0, disp1, T_src_trg_gt_np = euroc_dataset[index]
        images = [
            image_to_tensor(img0_L).to("cuda"),
            image_to_tensor(img1_L).to("cuda")
        ]
        disp0 = disp0.to("cuda")
        disp1 = disp1.to("cuda")
                
        print("Setting up stereo pipeline...")
        # Set up config
        config_path = ROOT / "configs" / "stereo_pipeline.yaml"
        config = load_config(config_path)
        # Turn on debug mode to visualize intermediate results
        config.debug = True
        # Get stereo camera config from dataset
        config.stereo_camera_config = euroc_preproc.get_stereo_cam_config()
        # Set up the stereo pipeline
        pipeline = StereoPipeline(config)
        # Initialize with ground truth transform
        print("Running stereo pipeline...")
        output = pipeline.forward(
            images=images,
            disparities=[disp0, disp1],
            T_init=T_src_trg_gt_np,  # initial guess for the relative transform
        )
        # Check that the estimated transform is close to the ground truth
        T_src_trg = Transformation(T_ba=output.relative_transform)
        T_src_trg_gt = Transformation(T_ba=T_src_trg_gt_np)
        xi_error = (T_src_trg.inverse() @ T_src_trg_gt).vec()
        trans_error = np.linalg.norm(xi_error[:3])
        rot_error = np.linalg.norm(xi_error[3:])
        print(f"Estimated transform:\n{T_src_trg.matrix()}")
        print(f"Ground truth transform:\n{T_src_trg_gt.matrix()}")
        print(
            f"Translation error: {trans_error:.4f} m, Rotation error: {rot_error:.4f} rad"
        )

        if plot:
            # Retrieve 3D keypoints and inliers/outliers
            kpt_3D_0 = output.debug_info.keypoints_3D[0].cpu().numpy()
            kpt_3D_1 = output.debug_info.keypoints_3D[1].cpu().numpy()
            inliers = output.debug_info.inliers.cpu().numpy()
            # Transform kpt_3D_1 to the frame of kpt_3D_0 using the estimated transform
            kpt_3D_1_in_0 = T_src_trg.matrix() @ kpt_3D_1  # (4, N)

            # Plot 2D Matches
            plot_outliers = False
            axes = viz2d.plot_images([img0_L, img1_L])
            keypoints_2D = output.debug_info.keypoints_2D
            viz2d.plot_matches(
                keypoints_2D[0].T, keypoints_2D[1].T, color="lime", lw=0.2
            )
            # Plot 3D Matches with no correction
            fig, ax = plt.subplots(
                1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"}
            )
            plot_pointclouds(
                kpt_3D_0,
                kpt_3D_1,
                inliers,
                ax[0],
                title="3D Keypoints (No Correction)",
                plot_outliers=plot_outliers,
            )
            # Plot 3D Matches with correction
            plot_pointclouds(
                kpt_3D_0,
                kpt_3D_1_in_0,
                inliers,
                ax[1],
                title="3D Keypoints (Transformed Frame 1 to Frame 0)",
                plot_outliers=plot_outliers,
            )
            plt.show()

if __name__ == "__main__":
    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    run_stereo_pipeline(index=2000, frame_skip=10, plot=True)
