"""
This script is used to compare data association methods on the Euroc dataset. It retrieves a rectified stereo pair, computes disparities, and runs the stereo pipeline to estimate the relative transform between the two frames. It also provides options to visualize the 2D and 3D matches, as well as the estimated and ground truth transforms.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
import os
from pylgmath import Transformation

# Ensure plotting uses the desired X display (useful in headless CI/devcontainer)
os.environ["DISPLAY"] = ":32"

from stereo_loc.EurocPreprocess import EurocPreprocess
from stereo_loc.DataAssociationBlocks import DataAssociationMethod
from stereo_loc.StereoPipeline import StereoPipeline, load_config
from utils.stereo_camera_model import (
    get_disparity,
)

from lightglue import viz2d
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def get_euroc_data():
    """Instantiate a EurocDataset, retrieve a rectified stereo pair and convert to torch tensors."""

    # Expected default dataset location inside the experiments tree
    default_root = ROOT / "data" / "Euroc" / "MH_01_easy"
    if not default_root.exists():
        raise (f"Euroc dataset not found at {default_root}")

    ds = EurocPreprocess(default_root)
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


def _run_pipeline_for_method(index0, index1, method: DataAssociationMethod):
    print(f"Collecting image data and generating disparities for {method.value}...")
    ds = get_euroc_data()
    timestamp0 = ds.cam0.timestamps[index0]
    timestamp1 = ds.cam1.timestamps[index1]
    T_01 = ds.get_relative_transform(timestamp0, timestamp1, camera_frame=True)
    im0_L_t, im0_R_t, im0_L, im0_R = get_euroc_stereo_image(ds, timestamp0)
    im1_L_t, im1_R_t, im1_L, im1_R = get_euroc_stereo_image(ds, timestamp1)
    images = [im0_L_t.float(), im1_L_t.float()]
    disp0 = get_disparity(im0_L, im0_R).float().to("cuda")
    disp1 = get_disparity(im1_L, im1_R).float().to("cuda")

    print("Setting up stereo pipeline...")
    config_path = ROOT / "configs" / "test_config.yaml"
    config = load_config(config_path)
    config.debug = True
    config.stereo_camera_config = ds.get_stereo_cam_config()
    config.data_association_config.method = method

    pipeline = StereoPipeline(config)
    print(f"Running stereo pipeline with {method.value}...")
    output = pipeline.forward(
        images=images,
        disparities=[disp0.float(), disp1.float()],
        T_init=T_01,
    )

    T_src_trg = Transformation(T_ba=output.relative_transform)

    return {
        "method": method,
        "output": output,
        "transform": T_src_trg,
        "inliers": output.debug_info.inliers.cpu().numpy(),
        "kpt_3d_0": output.debug_info.keypoints_3D[0].cpu().numpy(),
        "kpt_3d_1": output.debug_info.keypoints_3D[1].cpu().numpy(),
        "M": output.debug_info.M,
        "T_01": T_01,
    }


def plot_inlier_matches_3d(
    kpt_3D_0,
    kpt_3D_1,
    inliers_clipper,
    inliers_sdp,
    ax=None,
    title="3D Matches (No Correction)",
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    inliers_common = inliers_clipper & inliers_sdp
    all_inliers = inliers_clipper | inliers_sdp

    for i in range(inliers_common.shape[0]):
        if inliers_common[i]:
            line_color = "green"
        elif inliers_clipper[i]:
            line_color = "red"
        elif inliers_sdp[i]:
            line_color = "orange"
        else:
            continue

        ax.plot(
            [kpt_3D_0[0, i], kpt_3D_1[0, i]],
            [kpt_3D_0[1, i], kpt_3D_1[1, i]],
            [kpt_3D_0[2, i], kpt_3D_1[2, i]],
            c=line_color,
            linewidth=1.0,
            alpha=0.85,
        )

    ax.scatter(
        kpt_3D_0[0, all_inliers],
        kpt_3D_0[1, all_inliers],
        kpt_3D_0[2, all_inliers],
        c="blue",
        s=3,
        label="frame 0",
        alpha=0.7,
    )
    ax.scatter(
        kpt_3D_1[0, all_inliers],
        kpt_3D_1[1, all_inliers],
        kpt_3D_1[2, all_inliers],
        c="red",
        s=3,
        label="frame 1",
        alpha=0.7,
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    return ax


def print_method_info(result):
    method = result["method"]
    print(f"\nMethod: {method.value}")
    # Print Solution Cost
    M = result["M"]
    soln = result["output"].debug_info.da_soln
    cost = soln.T @ M @ soln
    # print error
    T_src_trg = result["transform"]
    T_src_trg_gt = Transformation(T_ba=result["T_01"])
    xi_error = (T_src_trg.inverse() @ T_src_trg_gt).vec()
    trans_error = np.linalg.norm(xi_error[:3])
    rot_error = np.linalg.norm(xi_error[3:])
    print(
        f"Certification result (association) ({method.value}): {result['output'].data_association_certified}"
    )
    print(
        f"Certification result (registration) ({method.value}): {result['output'].registration_certified}"
    )
    print(f"data association cost ({method.value}): {cost.item():.4f}")
    print(
        f"Number of inliers ({method.value}): {result['inliers'].sum()} / {len(result['inliers'])}"
    )
    print(f"Estimated transform ({method.value}):\n{T_src_trg.matrix()}")
    print(f"Ground truth transform:\n{T_src_trg_gt.matrix()}")
    print(
        f"Translation error ({method.value}): {trans_error:.4f} m, Rotation error ({method.value}): {rot_error:.4f} rad"
    )


def compare_with_sdp(index0=1000, index1=1030, plot=False):
    with torch.no_grad():
        clipper = _run_pipeline_for_method(
            index0, index1, DataAssociationMethod.CLIPPER
        )
        clipper_sdp = _run_pipeline_for_method(
            index0, index1, DataAssociationMethod.SDP
        )

        print_method_info(clipper)
        print_method_info(clipper_sdp)

        # Check that the same keypoints were matched by both methods
        keypoints_2D_clip = clipper["output"].debug_info.keypoints_2D
        keypoints_2D_sdp = clipper_sdp["output"].debug_info.keypoints_2D
        assert torch.allclose(keypoints_2D_clip, keypoints_2D_sdp)

        if plot:
            inliers_clip = clipper["inliers"]
            inliers_sdp = clipper_sdp["inliers"]
            if len(inliers_clip) != len(inliers_sdp):
                raise ValueError(
                    "CLIPPER and SDP produced different numbers of matched keypoints; cannot compare inlier masks directly."
                )

            common_inliers = inliers_clip & inliers_sdp

            plot_inlier_matches_3d(
                clipper["kpt_3d_0"],
                clipper["kpt_3d_1"],
                clipper["inliers"],
                clipper_sdp["inliers"],
                title="3D matches with no correction (CLIPPER)",
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

    compare_with_sdp(index0=1000, index1=1010, plot=True)
