from dataclasses import dataclass
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
from stereo_loc.EurocDataloader import EurocDataset, collate_skip_none

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
    experiment_name: str = "default_experiment"
    override_path: Path | None = None
    dataset_path: Path | None = None
    frame_interval: int = 1
    interval: Tuple[int, int] | None = None
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
    pipeline_cgf = load_config(default_pipeline_cfg_path)
    # Load any overrides specified in the experiment config
    pipeline_cgf = load_pipeline_overrides(pipeline_cgf, cfg.override_path)
    # Get stereo camera config from dataset
    pipeline_cgf.stereo_camera_config = euroc_preprocess.get_stereo_cam_config()
    # Initialize the stereo pipeline
    pipeline = StereoPipeline(pipeline_cgf)
    # Restrict to interval if specified
    if cfg.interval is not None:
        start, end = cfg.interval
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
        xi_error = (T_src_trg.inverse() @ T_src_trg_gt).vec()
        trans_error = np.linalg.norm(xi_error[:3])
        rot_error = np.linalg.norm(xi_error[3:])
        # Store results
        output_data.append(
            dict(
                index=idx,
                timestep=timestep,
                time_interval=time_interval,
                trans_error=trans_error,
                rot_error=rot_error,
                cert_da=output.data_association_certified,
                cert_reg=output.registration_certified,
            )
        )

    # Convert data to dataframe
    df = pd.DataFrame(output_data)
    output_csv_path = (
        ROOT / "results" / "stereo_loc" / f"{cfg.experiment_name}_results.csv"
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    exp_config = load_experiment_config(
        ROOT / "configs" / "stereo_experiments" / "test.yaml"
    )
    run_experiment(exp_config)
