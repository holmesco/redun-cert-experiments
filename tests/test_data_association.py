import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
config_file = ROOT / "configs" / "test_config.yaml"

from stereo_loc.DataAssociationBlocks import DataAssociationBlock
from stereo_loc.StereoPipeline import load_config
from fixtures import bunny_stereo_synthetic  # noqa: F401


def test_clipper_block(bunny_stereo_synthetic):

    points_c1 = bunny_stereo_synthetic["points_1"]
    points_c2 = bunny_stereo_synthetic["points_2"]
    n_outliers = bunny_stereo_synthetic["n_outliers"]

    # Instantiate CLIPPER block
    config = load_config(config_file)
    data_association_config = config.data_association_config
    data_association_config.invariant_epsilon = 0.002
    data_association_config.invariant_sigma = 0.001
    clipper_block = DataAssociationBlock(data_association_config)
    inliers, soln = clipper_block.run_clipper(
        torch.from_numpy(points_c1.T).float().to("cpu"),  # (4,N)
        torch.from_numpy(points_c2.T).float().to("cpu"),  # (4,N)
    )
    # assert that the original sampled points are inliers and the synthetic outliers are rejected
    assert (
        torch.sum(inliers) >= points_c1.shape[0] - n_outliers
    ), f"Expected at least {points_c1.shape[0] - n_outliers} inliers, but got {torch.sum(inliers)} inliers out of {points_c1.shape[0]} total points"
    # Test certification
    result = clipper_block.certify_solution(soln=soln, check_constraints=True)
    assert result.certified, "Certifier could not certify solution"


def test_inliers_to_soln(bunny_stereo_synthetic):
    points_c1 = bunny_stereo_synthetic["points_1"]
    points_c2 = bunny_stereo_synthetic["points_2"]
    n_outliers = bunny_stereo_synthetic["n_outliers"]

    # Instantiate CLIPPER block
    config = load_config(config_file)
    data_association_config = config.data_association_config
    data_association_config.invariant_epsilon = 0.02
    data_association_config.invariant_sigma = 0.01
    clipper_block = DataAssociationBlock(data_association_config)
    inliers, soln = clipper_block.run_clipper(
        torch.from_numpy(points_c1.T).float().to("cpu"),  # (4,N)
        torch.from_numpy(points_c2.T).float().to("cpu"),  # (4,N)
    )
    soln_from_inliers, cost = clipper_block.inliers_to_solution(inliers=inliers)
    # assert that the solutions are the same
    assert np.linalg.allclose(
        soln, soln_from_inliers, atol=1e-7
    ), "Solutions from run_clipper and inliers_to_soln do not match"


def test_clipper_block_threshold(bunny_stereo_synthetic):
    points_c1 = bunny_stereo_synthetic["points_1"]
    points_c2 = bunny_stereo_synthetic["points_2"]
    n_outliers = bunny_stereo_synthetic["n_outliers"]

    # Instantiate CLIPPER block
    config = load_config(config_file)
    data_association_config = config.data_association_config
    data_association_config.invariant_epsilon = 0.02
    data_association_config.invariant_sigma = 0.01
    data_association_config.unweighted = True  # Enable thresholding

    clipper_block = DataAssociationBlock(data_association_config)
    inliers, x = clipper_block.run_clipper(
        torch.from_numpy(points_c1.T).float().to("cpu"),  # (4,N)
        torch.from_numpy(points_c2.T).float().to("cpu"),  # (4,N)
    )

    assert (
        torch.sum(inliers) >= points_c1.shape[0] - n_outliers
    ), f"Expected at least {points_c1.shape[0] - n_outliers} inliers, but got {torch.sum(inliers)} inliers out of {points_c1.shape[0]} total points"
    # Test certification
    result = clipper_block.certify_solution(inliers=inliers, check_constraints=True)
    assert result.certified, "Certifier could not certify solution"


def test_clipper_sdp(bunny_stereo_synthetic):

    points_c1 = bunny_stereo_synthetic["points_1"]
    points_c2 = bunny_stereo_synthetic["points_2"]
    n_outliers = bunny_stereo_synthetic["n_outliers"]

    # Instantiate CLIPPER block
    config = load_config(config_file)
    data_association_config = config.data_association_config
    data_association_config.invariant_epsilon = 0.002
    data_association_config.invariant_sigma = 0.001
    clipper_block = DataAssociationBlock(data_association_config)
    # Run the CLIPPER SDP block to get inliers and the solution u
    inliers, u = clipper_block.run_sdp(
        torch.from_numpy(points_c1.T).float().to("cpu"),  # (4,N)
        torch.from_numpy(points_c2.T).float().to("cpu"),  # (4,N)
    )
    # assert that the original sampled points are inliers and the synthetic outliers are rejected
    assert (
        torch.sum(inliers) >= points_c1.shape[0] - n_outliers
    ), f"Expected at least {points_c1.shape[0] - n_outliers} inliers, but got {torch.sum(inliers)} inliers out of {points_c1.shape[0]} total points"
    # Test certification
    result = clipper_block.certify_solution(soln=u, check_constraints=True)
    assert result.certified, "Certifier could not certify solution"
