import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
config_file = ROOT / "configs" / "stereo_pipeline" / "test_config.yaml"

from stereo_loc.DataAssociationBlocks import DataAssociationBlock, estimate_pose_svd
from stereo_loc.StereoPipeline import load_config
from fixtures import bunny_stereo_synthetic  # noqa: F401


def test_data_association(bunny_stereo_synthetic):

    points_c1 = bunny_stereo_synthetic["points_1"]
    points_c2 = bunny_stereo_synthetic["points_2"]
    n_outliers = bunny_stereo_synthetic["n_outliers"]

    # Instantiate CLIPPER block
    config = load_config(config_file)
    data_association_config = config.data_association_config
    data_association_config.invariant_epsilon = 0.002
    data_association_config.invariant_sigma = 0.001
    data_association = DataAssociationBlock(data_association_config)
    inliers, soln = data_association.run_clipper(
        torch.from_numpy(points_c1.T).float().to("cpu"),  # (4,N)
        torch.from_numpy(points_c2.T).float().to("cpu"),  # (4,N)
    )
    # assert that the original sampled points are inliers and the synthetic outliers are rejected
    assert (
        torch.sum(inliers) >= points_c1.shape[0] - n_outliers
    ), f"Expected at least {points_c1.shape[0] - n_outliers} inliers, but got {torch.sum(inliers)} inliers out of {points_c1.shape[0]} total points"
    # Test certification
    result = data_association.certify_solution(soln, check_constraints=True)
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
    data_association = DataAssociationBlock(data_association_config)
    inliers, soln = data_association.run_clipper(
        torch.from_numpy(points_c1.T).float().to("cpu"),  # (4,N)
        torch.from_numpy(points_c2.T).float().to("cpu"),  # (4,N)
    )
    soln_from_inliers, cost = data_association.inliers_to_solution(inliers=inliers)
    # assert that the solutions are the same
    assert np.linalg.allclose(
        soln, soln_from_inliers, atol=1e-7
    ), "Solutions from run_clipper and inliers_to_soln do not match"


def test_data_association_threshold(bunny_stereo_synthetic):
    points_c1 = bunny_stereo_synthetic["points_1"]
    points_c2 = bunny_stereo_synthetic["points_2"]
    n_outliers = bunny_stereo_synthetic["n_outliers"]

    # Instantiate CLIPPER block
    config = load_config(config_file)
    data_association_config = config.data_association_config
    data_association_config.invariant_epsilon = 0.02
    data_association_config.invariant_sigma = 0.01
    data_association_config.unweighted = True  # Enable thresholding

    data_association = DataAssociationBlock(data_association_config)
    inliers, x = data_association.run_clipper(
        torch.from_numpy(points_c1.T).float().to("cpu"),  # (4,N)
        torch.from_numpy(points_c2.T).float().to("cpu"),  # (4,N)
    )

    assert (
        torch.sum(inliers) >= points_c1.shape[0] - n_outliers
    ), f"Expected at least {points_c1.shape[0] - n_outliers} inliers, but got {torch.sum(inliers)} inliers out of {points_c1.shape[0]} total points"
    # Test certification

    result = data_association.certify_solution(x, check_constraints=True)
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
    data_association = DataAssociationBlock(data_association_config)
    # Run the CLIPPER SDP block to get inliers and the solution u
    inliers, u = data_association.run_sdp(
        torch.from_numpy(points_c1.T).float().to("cpu"),  # (4,N)
        torch.from_numpy(points_c2.T).float().to("cpu"),  # (4,N)
    )
    # assert that the original sampled points are inliers and the synthetic outliers are rejected
    assert (
        torch.sum(inliers) >= points_c1.shape[0] - n_outliers
    ), f"Expected at least {points_c1.shape[0] - n_outliers} inliers, but got {torch.sum(inliers)} inliers out of {points_c1.shape[0]} total points"
    # Test certification
    result = data_association.certify_solution(u, check_constraints=True)
    assert result.certified, "Certifier could not certify solution"


def test_svd_estimator(bunny_stereo_synthetic):
    points_c1 = bunny_stereo_synthetic["points_1"]
    points_c2 = bunny_stereo_synthetic["points_2"]
    T_12 = bunny_stereo_synthetic["T_12"]
    T_trg_src_gt = torch.from_numpy(np.linalg.inv(T_12))

    # Remove outliers from the points
    n_outliers = bunny_stereo_synthetic["n_outliers"]
    points_c1 = points_c1[: points_c1.shape[0] - n_outliers, :]
    points_c2 = points_c2[: points_c2.shape[0] - n_outliers, :]

    # Run the SVD estimator to get the transformation matrix
    T_trg_src = estimate_pose_svd(
        torch.from_numpy(points_c1.T).float().to("cpu"),  # (4,N)
        torch.from_numpy(points_c2.T).float().to("cpu"),  # (4,N)
    )

    assert T_trg_src.shape == (
        4,
        4,
    ), "SVD estimator did not return a valid transformation matrix"
    assert np.allclose(
        T_trg_src.numpy(), T_trg_src_gt, atol=1e-6
    ), "SVD estimator did not return the correct transformation matrix"


def test_ransac_cert(bunny_stereo_synthetic):
    points_c1 = bunny_stereo_synthetic["points_1"]
    points_c2 = bunny_stereo_synthetic["points_2"]
    n_outliers = bunny_stereo_synthetic["n_outliers"]

    # Instantiate CLIPPER block
    config = load_config(config_file)
    data_association_config = config.data_association_config
    data_association_config.invariant_epsilon = 0.002
    data_association_config.invariant_sigma = 0.001
    data_association = DataAssociationBlock(data_association_config)
    # Run the CLIPPER SDP block to get inliers and the solution u
    inliers, x_torch, cost = data_association.run_ransac(
        torch.from_numpy(points_c1.T).float().to("cpu"),  # (4,N)
        torch.from_numpy(points_c2.T).float().to("cpu"),  # (4,N)
    )
    # assert that the original sampled points are inliers and the synthetic outliers are rejected
    assert (
        torch.sum(inliers) >= points_c1.shape[0] - n_outliers
    ), f"Expected at least {points_c1.shape[0] - n_outliers} inliers, but got {torch.sum(inliers)} inliers out of {points_c1.shape[0]} total points"
    # Test certification
    x = x_torch.cpu().numpy()
    result = data_association.certify_solution(x, check_constraints=True)
    assert result.certified, "Certifier could not certify solution"
