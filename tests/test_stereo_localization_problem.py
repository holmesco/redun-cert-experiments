import numpy as np
import pytest
import torch

import gtsam

from mat_weight_loc.one_pose_stereo_loc import (
    sim_single_pose_localization,
    SinglePoseStereoLocalization,
)
from utils.lie_algebra import se3_exp


@pytest.fixture(scope="module", autouse=True)
def double_precision():
    """The certifier and factor graph run in double precision.

    Set the default dtype here rather than at import time so that other test
    modules (which build single-precision torch models) are not affected.
    """
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous_dtype)


def _pose_error(T_est: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
    """Lie algebra error between two 4x4 homogeneous transforms."""
    return gtsam.Pose3(np.asarray(T_est)).logmap(gtsam.Pose3(np.asarray(T_ref)))


@pytest.fixture(scope="module")
def stereo_problem(double_precision):
    return sim_single_pose_localization(
        N_map=50,
        seed=0,
    )


class TestFactorGraphOptimization:
    def test_converges_from_ground_truth_initialization(
        self, stereo_problem: SinglePoseStereoLocalization
    ):
        T_est, info = stereo_problem.solve_factor_graph(
            stereo_problem.T_trg_src,
            verbose=True,
        )
        err = _pose_error(T_est, stereo_problem.T_trg_src)
        np.testing.assert_allclose(err, np.zeros(6), atol=1e-7)
        assert info["cost"] < 1e-10, f"Residual cost too large: {info['cost']}"

    def test_converges_from_perturbed_initialization(
        self, stereo_problem: SinglePoseStereoLocalization
    ):
        pert = 0.5
        xi_pert = torch.tensor([[pert, pert, pert, pert, pert, pert]])
        T_pert = se3_exp(xi_pert)[0].numpy()
        T_init = T_pert @ stereo_problem.T_trg_src

        T_est, info = stereo_problem.solve_factor_graph(
            T_init,
            verbose=True,
        )
        err = _pose_error(T_est, stereo_problem.T_trg_src)
        np.testing.assert_allclose(err, np.zeros(6), atol=1e-7)
        assert info["cost"] < 1e-10, f"Residual cost too large: {info['cost']}"


class TestCertification:
    def test_sdp_solution(self, stereo_problem: SinglePoseStereoLocalization):
        # Solve SDP
        X, _ = stereo_problem.solve_sdp(verbose=True)
        # Check rank-1 solution
        eigenvalues = np.sort(np.linalg.eigvalsh(X))[::-1]
        assert (
            eigenvalues[0] / eigenvalues[1] > 1e6
        ), f"Eigenvalue ratio: {eigenvalues[0] / eigenvalues[1]}"
        # Extract the rank-1 factor, normalized by the homogenization variable
        x_sdp = X[:, 0] / np.sqrt(X[0, 0])
        values = stereo_problem.values_from_vector(x_sdp)
        T_trg_src_sdp = values.atPose3(gtsam.Symbol("x", 0).key()).matrix()
        # Check that solution is close to ground truth
        err = _pose_error(T_trg_src_sdp, stereo_problem.T_trg_src)
        assert (
            np.linalg.norm(err) < 1e-6
        ), f"SDP solution not close to ground truth: {err}"
        print(f"SDP solution close to ground truth: {np.linalg.norm(err)}")
        # Certify the extracted solution
        result = stereo_problem.certify_single_pose_solution(
            T_trg_src_sdp, verbose=True
        )
        assert bool(result.certified)

    def test_factor_graph_solution(self, stereo_problem: SinglePoseStereoLocalization):
        T_est, _ = stereo_problem.solve_factor_graph(
            stereo_problem.T_trg_src,
            verbose=False,
        )
        err = _pose_error(T_est, stereo_problem.T_trg_src)
        assert (
            np.linalg.norm(err) < 1e-6
        ), f"Factor graph solution not close to ground truth: {err}"
        print(f"Factor graph solution close to ground truth: {np.linalg.norm(err)}")
        result = stereo_problem.certify_single_pose_solution(T_est, verbose=True)
        assert bool(result.certified)
