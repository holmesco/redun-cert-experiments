import numpy as np
import pytest
import torch

import gtsam

from mat_weight_loc.stereo_loc import (
    sim_single_pose_localization,
    SinglePoseStereoLocalization,
)
from utils.lie_algebra import se3_exp, se3_inv, se3_log


# Default dtype
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _assert_pose_close(T_est: torch.Tensor, T_ref: torch.Tensor, atol: float) -> None:
    diff = se3_log(se3_inv(T_est.cpu()).bmm(T_ref.cpu())).numpy()
    np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=atol)


@pytest.fixture(scope="module")
def stereo_problem():
    return sim_single_pose_localization(
        N_map=50,
        seed=0,
    )


class TestFactorGraphOptimization:
    def test_converges_from_ground_truth_initialization(
        self, stereo_problem: SinglePoseStereoLocalization
    ):
        T_est, _ = stereo_problem.solve_factor_graph(
            stereo_problem.T_trg_src[0].cpu().numpy(),
            verbose=True,
        )
        T_est = torch.from_numpy(T_est[None, :, :])
        _assert_pose_close(T_est, stereo_problem.T_trg_src, atol=1e-7)

    def test_converges_from_perturbed_initialization(
        self, stereo_problem: SinglePoseStereoLocalization
    ):
        pert = 0.5
        xi_pert = torch.tensor(
            [[pert, pert, pert, pert, pert, pert]],
            dtype=stereo_problem.T_trg_src.dtype,
            device=stereo_problem.T_trg_src.device,
        )
        T_pert = se3_exp(xi_pert)
        T_init = T_pert.bmm(stereo_problem.T_trg_src)

        T_est, _ = stereo_problem.solve_factor_graph(
            T_init[0].cpu().numpy(),
            verbose=True,
        )
        T_est = torch.from_numpy(T_est[None, :, :])
        _assert_pose_close(T_est, stereo_problem.T_trg_src, atol=1e-8)


class TestCertification:
    def test_sdp_solution(self, stereo_problem: SinglePoseStereoLocalization):
        # Solve SDP
        X, _ = stereo_problem.solve_sdp(verbose=True)
        # Check rank-1 solution
        eigenvalues = np.linalg.eigvalsh(X)
        eigenvalues = np.sort(eigenvalues)[::-1]
        assert (
            eigenvalues[0] / eigenvalues[1] > 1e6
        ), f"Eigenvalue ratio: {eigenvalues[0] / eigenvalues[1]}"
        # Check that solution is close to ground truth
        values = stereo_problem.values_from_vector(X[:, 0])
        key = gtsam.Symbol("x", 0).key()
        T_trg_src_sdp = values.atPose3(key)
        T_trg_src_gt = gtsam.Pose3(stereo_problem.T_trg_src)
        err = T_trg_src_sdp.logmap(T_trg_src_gt)
        assert (
            np.linalg.norm(err) < 1e-6
        ), f"SDP solution not close to ground truth: {err}"
        print(f"SDP solution close to ground truth: {np.linalg.norm(err)}")
        result = stereo_problem.certify_solution(T_trg_src_sdp.matrix(), verbose=True)
        assert bool(result.certified)

    def test_factor_graph_solution(self, stereo_problem: SinglePoseStereoLocalization):
        T_est, _ = stereo_problem.solve_factor_graph(
            stereo_problem.T_trg_src,
            verbose=False,
        )
        T_trg_est = gtsam.Pose3(T_est)
        T_trg_src_gt = gtsam.Pose3(stereo_problem.T_trg_src)
        err = T_trg_est.logmap(T_trg_src_gt)
        assert (
            np.linalg.norm(err) < 1e-6
        ), f"SDP solution not close to ground truth: {err}"
        print(f"SDP solution close to ground truth: {np.linalg.norm(err)}")
        result = stereo_problem.certify_solution(T_trg_est.matrix(), verbose=True)
        assert bool(result.certified)
