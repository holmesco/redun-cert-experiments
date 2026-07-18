from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, List
from pathlib import Path
import os
import numpy as np
from enum import Enum
from itertools import islice

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pylgmath import Transformation
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
import pandas as pd
import matplotlib.pyplot as plt
from lightglue import viz2d

# Ensure plotting uses the desired X display (useful in headless CI/devcontainer)
os.environ["DISPLAY"] = ":32"

from stereo_loc.EurocPreprocess import EurocPreprocess
from stereo_loc.StereoPipeline import StereoPipeline, StereoPipelineConfig, load_config
from stereo_loc.EurocDataloader import EurocDataset

ROOT = Path(__file__).resolve().parents[2]
default_pipeline_cfg_path = (
    ROOT / "configs" / "stereo_pipeline" / "stereo_pipeline_default.yaml"
)


class PoseInitializationMethod(Enum):
    IDENTITY = 1
    GROUND_TRUTH = 2
    RANDOM = 3


@dataclass
class StereoPipelineExperimentConfig:
    # Name of the experiment, used for saving results
    experiment_name: str = "default_experiment"
    # Path to the YAML file containing overrides for the stereo pipeline configuration
    override_path: Path | None = None
    # Path to the dataset to be used for the experiment
    dataset_path: Path | None = None
    # Intervals between frames in the dataset for registration. The experiment is
    # run (with a freshly defined data loader) for each frame interval.
    frame_interval: List[int] = field(default_factory=lambda: [1])
    # Limits on the indices of the dataset to be used for the experiment (inclusive). If None, use the entire dataset.
    index_bounds: Tuple[int, int] | None = None
    # Method for initializing the pose for the registration algorithm
    pose_init: PoseInitializationMethod = PoseInitializationMethod.IDENTITY
    # Whether to save the results of the experiment
    save_results: bool = True
    # Seed for random number generation, useful for reproducibility
    random_seed: int = 42
    # Whether to shuffle the dataset before processing
    shuffle: bool = False
    # Number of samples to use for the experiment. If None, use the entire dataset.
    num_samples: int | None = None
    # Plotting options for visualizing the results
    plot: bool = False
    # Sweep bounds for the invariant noise thresholds. The multiplier is applied to
    # the data association invariant_sigma / invariant_epsilon, and the experiment
    # is run for each multiplier value.
    invariant_mult_min: float = 1.0
    invariant_mult_max: float = 1.0
    # Number of values for the threshold multiplier to sweep over (log spaced between min and max).
    invariant_mult_num: int = 1


def load_experiment_config(config_path: Path) -> StereoPipelineExperimentConfig:
    # Start with defaults from dataclass
    config = OmegaConf.structured(StereoPipelineExperimentConfig)

    # Merge overrides if provided
    if config_path:
        overrides = OmegaConf.load(ROOT / config_path)
        config = OmegaConf.merge(config, overrides)

    return OmegaConf.to_object(config)


def load_pipeline_overrides(config: StereoPipelineConfig, override_path: Path):
    if override_path:
        overrides = OmegaConf.load(ROOT / override_path)
        config = OmegaConf.merge(config, overrides)
    return OmegaConf.to_object(config)


def get_pose_initialization(
    pose_init_method: PoseInitializationMethod, T_src_trg_gt: np.ndarray
) -> np.ndarray:
    if pose_init_method == PoseInitializationMethod.IDENTITY:
        return np.eye(4)
    elif pose_init_method == PoseInitializationMethod.GROUND_TRUTH:
        return T_src_trg_gt
    elif pose_init_method == PoseInitializationMethod.RANDOM:
        # Generate a random transformation matrix
        random_transform = np.eye(4)
        random_transform[:3, 3] = np.random.uniform(-1, 1, size=3)  # Random translation
        random_rotation = Rotation.random().as_matrix()  # Random rotation matrix
        random_transform[:3, :3] = random_rotation
        return random_transform
    else:
        raise ValueError(f"Unknown pose initialization method: {pose_init_method}")


def run_experiment(cfg: StereoPipelineExperimentConfig):
    # Set all seeds for reproducibility
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)

    # Load dataset preprocessor (frame-interval independent)
    euroc_preprocess = EurocPreprocess(ROOT / cfg.dataset_path)
    # Load default configuration
    pipeline_cfg = load_config(default_pipeline_cfg_path)
    # Load any overrides specified in the experiment config
    pipeline_cfg = load_pipeline_overrides(pipeline_cfg, cfg.override_path)
    # Get stereo camera config from dataset
    pipeline_cfg.stereo_camera_config = euroc_preprocess.get_stereo_cam_config()
    # If plotting then set debug mode
    pipeline_cfg.debug = cfg.plot
    # Initialize the stereo pipeline
    pipeline = StereoPipeline(pipeline_cfg)
    # Base invariant values, scaled by each invariant multiplier below.
    base_invariant_sigma = pipeline_cfg.data_association_config.invariant_sigma
    base_invariant_epsilon = pipeline_cfg.data_association_config.invariant_epsilon

    # Generate array of multiplier values for the invariant values (log spaced).
    invariant_mults = np.logspace(
        np.log10(cfg.invariant_mult_min),
        np.log10(cfg.invariant_mult_max),
        cfg.invariant_mult_num,
    )

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Run the experiment, sweeping over frame intervals and invariant multipliers.
    output_data = []
    for frame_interval in cfg.frame_interval:
        # Redefine the dataset / data loader for this frame interval.
        euroc_dataset = EurocDataset(euroc_preprocess, frame_interval=frame_interval)
        # Restrict to bounds if specified
        if cfg.index_bounds is not None:
            start, end = cfg.index_bounds
            dataset = Subset(euroc_dataset, range(start, end + 1))
        else:
            dataset = euroc_dataset
        # Create a DataLoader for batch processing
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            batch_sampler=None,
            num_workers=0,
            shuffle=cfg.shuffle,
        )

        for invariant_mult in invariant_mults:
            # Reset CLIPPER using the scaled invariant values for this multiplier.
            pipeline.data_association.set_clipper(
                invariant_sigma=base_invariant_sigma * invariant_mult,
                invariant_epsilon=base_invariant_epsilon * invariant_mult,
            )
            # Restrict the number of samples per run if requested (islice yields a
            # one-shot iterator, so it is rebuilt for every sweep configuration).
            frames = (
                islice(dataloader, cfg.num_samples)
                if cfg.num_samples is not None
                else dataloader
            )
            for data in tqdm(
                frames,
                total=cfg.num_samples,
                desc=f"frame_interval={frame_interval}, inv_mult={invariant_mult:.3g}",
            ):
                if data is None:
                    continue  # Skip if the collate function returned None
                else:
                    (
                        idx,
                        timestep,
                        time_interval,
                        img0_L,
                        img1_L,
                        disp0,
                        disp1,
                        T_src_trg_gt,
                    ) = data
                # Add batch dimension and send to device
                images = [
                    img0_L.unsqueeze(0).to(device),
                    img1_L.unsqueeze(0).to(device),
                ]
                disp0 = disp0.unsqueeze(0).to(device)
                disp1 = disp1.unsqueeze(0).to(device)
                # Get pose initialization
                T_init = get_pose_initialization(cfg.pose_init, T_src_trg_gt)
                # Run model
                output = pipeline.forward(
                    images=images,
                    disparities=[disp0, disp1],
                    T_init=T_init,  # initial guess for the relative transform
                )
                # Check that the estimated transform is close to the ground truth
                T_src_trg = Transformation(T_ba=output.relative_transform)
                T_src_trg_gt = Transformation(T_ba=T_src_trg_gt)
                T_error = T_src_trg * T_src_trg_gt.inverse()
                err_trans = np.linalg.norm(T_error.r_ab_inb())
                err_rot = np.linalg.norm(
                    Rotation.from_matrix(T_error.C_ba()).as_rotvec()
                )
                delta_trans = np.linalg.norm(T_src_trg_gt.r_ab_inb())
                delta_rot = np.linalg.norm(
                    Rotation.from_matrix(T_src_trg_gt.C_ba()).as_rotvec()
                )

                # Store results
                output_data.append(
                    dict(
                        index=idx,
                        frame_interval=frame_interval,
                        inv_mult=invariant_mult,
                        timestep=timestep,
                        time_interval=time_interval,
                        err_trans=err_trans,
                        err_rot=err_rot,
                        delta_rot=delta_rot,
                        delta_trans=delta_trans,
                        cert_da=output.data_association_certified,
                        cert_reg=output.registration_certified,
                        cert_time_da=output.data_association_cert_time,
                        cert_time_reg=output.registration_cert_time,
                        num_inliers=output.num_inliers,
                        num_valid=output.num_valid,
                    )
                )

    if cfg.plot:
        # Retrieve 3D keypoints and inliers/outliers
        kpt_3D_0 = output.debug_info.keypoints_3D[0].cpu().numpy()
        kpt_3D_1 = output.debug_info.keypoints_3D[1].cpu().numpy()
        inliers = output.debug_info.inliers.cpu().numpy()
        # Transform kpt_3D_1 to the frame of kpt_3D_0 using the estimated transform
        kpt_3D_1_in_0 = T_src_trg.matrix() @ kpt_3D_1  # (4, N)

        # Plot 2D Matches
        plot_outliers = True
        axes = viz2d.plot_images([img0_L, img1_L])
        keypoints_2D = output.debug_info.keypoints_2D
        viz2d.plot_matches(keypoints_2D[0].T, keypoints_2D[1].T, color="lime", lw=0.2)
        # Plot 3D Matches with no correction
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})
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

    # Convert data to dataframe
    df = pd.DataFrame(output_data)
    if cfg.save_results:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M")
        run_dir = ROOT / "results" / "stereo_loc" / cfg.experiment_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(run_dir / "results.csv", index=False)
        OmegaConf.save(OmegaConf.structured(cfg), run_dir / "experiment.yaml")
        OmegaConf.save(
            OmegaConf.structured(pipeline_cfg), run_dir / "stereo_pipeline.yaml"
        )
    else:
        print("Experiment results:")
        print(df)


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
        c="magenta",
        s=3,
        label="frame 0",
        alpha=0.7,
    )

    ax.scatter(
        kpt_3D_2[0, inliers],
        kpt_3D_2[1, inliers],
        kpt_3D_2[2, inliers],
        c="blue",
        s=3,
        alpha=0.7,
        label="inliers (frame 1)",
    )
    if plot_outliers:
        ax.scatter(
            kpt_3D_0[0, outlier_mask],
            kpt_3D_0[1, outlier_mask],
            kpt_3D_0[2, outlier_mask],
            c="magenta",
            s=3,
            alpha=0.7,
            label="outliers (frame 1)",
        )
        ax.scatter(
            kpt_3D_2[0, outlier_mask],
            kpt_3D_2[1, outlier_mask],
            kpt_3D_2[2, outlier_mask],
            c="blue",
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

    # ax.set_title(title)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    ax.set_aspect("equal", adjustable="box")
    # No Background
    ax.set_facecolor("none")
    fig = plt.gcf()
    fig.patch.set_alpha(0.0)
    ax.axis("off")
    ax.grid(False)
    ax.view_init(elev=-90, azim=90, roll=180)

    return ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="test.yaml")
    args = parser.parse_args()

    exp_cfg_path = ROOT / "configs" / "stereo_experiments" / args.filename
    exp_config = load_experiment_config(exp_cfg_path)
    run_experiment(exp_config)
