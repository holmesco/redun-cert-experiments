from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List
from pathlib import Path
import os
import numpy as np
from enum import Enum

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pylgmath import Transformation
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
import pandas as pd

# Ensure plotting uses the desired X display (useful in headless CI/devcontainer)
os.environ["DISPLAY"] = ":32"

from stereo_loc.EurocPreprocess import EurocPreprocess
from stereo_loc.StereoPipeline import StereoPipeline, StereoPipelineConfig, load_config
from stereo_loc.EurocDataloader import EurocDataset

ROOT = Path(__file__).resolve().parents[1]
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
    # Interval between frames in the dataset for registration
    frame_interval: int = 1
    # Limits on the indices of the dataset to be used for the experiment (inclusive). If None, use the entire dataset.
    index_bounds: Tuple[int, int] | None = None
    # Method for initializing the pose for the registration algorithm
    pose_init: PoseInitializationMethod = PoseInitializationMethod.IDENTITY


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
    # Load Dataset
    euroc_preprocess = EurocPreprocess(ROOT / cfg.dataset_path)
    euroc_dataset = EurocDataset(euroc_preprocess, frame_interval=cfg.frame_interval)
    # Load default configuration
    pipeline_cfg = load_config(default_pipeline_cfg_path)
    # Load any overrides specified in the experiment config
    pipeline_cfg = load_pipeline_overrides(pipeline_cfg, cfg.override_path)
    # Get stereo camera config from dataset
    pipeline_cfg.stereo_camera_config = euroc_preprocess.get_stereo_cam_config()
    # Initialize the stereo pipeline
    pipeline = StereoPipeline(pipeline_cfg)
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
        shuffle=False,
    )
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Run the experiment
    output_data = []
    for data in tqdm(dataloader, desc="Processing frames"):
        if data is None:
            continue  # Skip if the collate function returned None
        else:
            idx, timestep, time_interval, img0_L, img1_L, disp0, disp1, T_src_trg_gt = (
                data
            )
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
        # Relative Rotation Error in radians
        rot_error = np.arccos((np.trace(T_src_trg_gt.C_ba().T @ T_src_trg.C_ba())-1)/2)
        trans_error = np.linalg.norm(T_src_trg_gt.r_ab_inb() - T_src_trg.r_ab_inb())
        rot_delta = np.arccos((np.trace(T_src_trg_gt.C_ba())-1)/2)
        trans_delta = np.linalg.norm(T_src_trg_gt.r_ab_inb())
        # Store results
        output_data.append(
            dict(
                index=idx,
                timestep=timestep,
                time_interval=time_interval,
                trans_error=trans_error,
                rot_error=rot_error,
                trans_delta=trans_delta,
                rot_delta=rot_delta,
                cert_da=output.data_association_certified,
                cert_reg=output.registration_certified,
                num_inliers=output.num_inliers,
            )
        )

    # Convert data to dataframe
    df = pd.DataFrame(output_data)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M")
    run_dir = ROOT / "results" / "stereo_loc" / cfg.experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(run_dir / "results.csv", index=False)
    OmegaConf.save(OmegaConf.structured(cfg), run_dir / "experiment.yaml")
    OmegaConf.save(OmegaConf.structured(pipeline_cfg), run_dir / "stereo_pipeline.yaml")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="test.yaml")
    args = parser.parse_args()

    exp_cfg_path = ROOT / "configs" / "stereo_experiments" / args.filename
    exp_config = load_experiment_config(exp_cfg_path)
    run_experiment(exp_config)
