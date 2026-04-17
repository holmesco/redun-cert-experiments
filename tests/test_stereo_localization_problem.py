import numpy as np
import pytest
import torch

from mat_weight_loc.stereo_loc import create_stereo_localization_problem
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
    return create_stereo_localization_problem(
        batch_size=1,
        N_map=50,
        device=_default_device(),
        seed=0,
    )

class TestFactorGraphOptimization:
    def test_converges_from_ground_truth_initialization(self, stereo_problem):
        T_est, _ = stereo_problem.solve_factor_graph(
            stereo_problem.T_trg_src[0].cpu().numpy(),
            verbose=True,
        )
        T_est = torch.from_numpy(T_est[None, :, :])
        _assert_pose_close(T_est, stereo_problem.T_trg_src, atol=1e-7)

    def test_converges_from_perturbed_initialization(self, stereo_problem):
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
    def test_sdp_solution_matches_ground_truth_and_is_certified(self, stereo_problem):
        _, _, T_trg_src_sdp = stereo_problem.certifier.solve_sdp(verbose=False)

        _assert_pose_close(T_trg_src_sdp, stereo_problem.T_trg_src, atol=1e-6)

        x_cand = stereo_problem.certifier.transform_to_x(T_trg_src_sdp)
        result = stereo_problem.certifier.certify_solution(x_cand[0])
        assert bool(result.certified)

    def test_factor_graph_solution_is_certified(self, stereo_problem):
        T_est, _ = stereo_problem.solve_factor_graph(
            stereo_problem.T_trg_src[0].cpu().numpy(),
            verbose=False,
        )
        T_est = torch.from_numpy(T_est[None, :, :])
        x_cand = stereo_problem.certifier.transform_to_x(T_est)
        x_cand = x_cand.detach().cpu().numpy()
        result = stereo_problem.certifier.certify_solution(x_cand[0])
        assert bool(result.certified)
