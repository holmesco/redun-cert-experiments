import torch
import numpy as np
import time

import gtsam
from gtsam.symbol_shorthand import X, S, T

from utils.stereo_camera_model import StereoCameraModel
from utils.stereo_utils import get_gt_setup
from utils.keypoint_tools import get_inv_cov_weights
from utils.lie_algebra import se3_exp

from mat_weight_loc.stereo_cert import StereoPoseCertifier
from mwcerts.cert_factor_graph import LocalizationFactorGraph
from ranktools import (
    AnalyticCenter,
    AnalyticCenterParams,
    LinearSolverType,
    LowRankPrecondMethod,
)


def get_default_cert_params():
    params = AnalyticCenterParams()
    params.verbose = True
    params.lin_solver = LinearSolverType.MFCG_LRP
    params.lin_solve_max_iter = 200
    params.lin_solve_tol = 1e-4
    params.delta_init = 1e-5
    params.rescale_lin_sys = False
    params.max_iter = 20
    # Early stopping local
    params.early_stop_angle = True
    params.max_angle = 1e-3
    # Preconditioner parameters
    params.lrp_params.tau = 1e-4
    params.lrp_params.method = LowRankPrecondMethod.SparseLDLT
    # Turn off perturbations:
    params.delta_min = 1e-8
    params.perturb_constraints = False
    params.perturb_cost = False
    params.adaptive_perturb = False
    params.cost_perturb = 1e-6
    return params


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


# Default dtype
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)


class SinglePoseStereoLocalization(LocalizationFactorGraph):
    def __init__(
        self,
        keypoints_3D_src: np.ndarray,
        keypoints_3D_trg: np.ndarray,
        weights: np.ndarray,
        inv_cov_weights: np.ndarray,
        T_s_v: np.ndarray | None = None,
        params: AnalyticCenterParams | None = None,
    ):

        super().__init__()

        assert (
            len(keypoints_3D_src.shape) == 2
        ), "keypoints_3D_src should have shape (3, N_map)"
        assert (
            len(keypoints_3D_trg.shape) == 2
        ), "keypoints_3D_trg should have shape (3, N_map)"
        assert (
            keypoints_3D_src.shape[1] == keypoints_3D_trg.shape[1]
        ), "keypoints_3D_src and keypoints_3D_trg should have the same shape"
        assert (
            weights.shape == keypoints_3D_src.shape[1:]
        ), "weights should have shape (N_map,)"
        assert T_s_v.shape == (4, 4), "T_s_v should have shape (4, 4)"

        if T_s_v is None:
            T_s_v = np.eye(4)
        self.T_s_v = T_s_v
        self.N_map = keypoints_3D_src.shape[1]
        self.T_trg_src = None

        # Add factors for keypoints
        self.add_fixed_keypoint_factors(
            pose_id=gtsam.Symbol("x", 0),
            keypoints_3D_src=keypoints_3D_src,
            keypoints_3D_trg=keypoints_3D_trg,
            weights=inv_cov_weights,
        )
        
        # Add constraints
        self.add_constraints()

        # Certifier Parameters
        if params is None:
            self.cert_params = get_default_cert_params()
        else:
            self.cert_params = params

    def get_random_inits(self, radius, N_batch=10, seed=0):
        """Generate random pose initializations similar to stereo_cal.get_random_inits."""
        set_seed(seed)
        r_v0s = []
        C_v0s = []

        for _ in range(N_batch):
            # random locations on sphere
            r_ = np.random.random((3, 1)) - 0.5
            r = radius * r_ / np.linalg.norm(r_)
            r_v0s += [r]

            # random orientation pointing at origin
            z = -r / np.linalg.norm(r)
            y = np.random.randn(3, 1)
            y = y - y.T @ z * z
            y = y / np.linalg.norm(y)
            x = np.cross(y[:, 0], z[:, 0])[:, None]
            C_v0s += [np.hstack([x, y, z]).T]

        r_v0s = np.stack(r_v0s)
        C_v0s = np.stack(C_v0s)

        return r_v0s, C_v0s

    def solve_factor_graph(self, T_init: np.ndarray, verbose:bool=False):
        """Solve the factor graph optimization problem starting from T_init."""
        # Build initial values
        initial_values = gtsam.Values()
        initial_values.insert(
            X(0), gtsam.Pose3(gtsam.Rot3(T_init[:3, :3]), gtsam.Point3(T_init[:3, 3]))
        )
        # Run optimization
        result, time = self.optimize_graph(initial_values, verbose=verbose)
        # Extract solution
        T_est = result.atPose3(gtsam.Symbol("x", 0).key()).matrix()
        return T_est, time

    def certify_solution(self, T_est: np.ndarray, verbose=False):

        # Get the vector version of the solution
        values = gtsam.Values()
        values.insert(
            gtsam.Symbol("x", 0).key(),
            gtsam.Pose3(gtsam.Rot3(T_est[:3, :3]), gtsam.Point3(T_est[:3, 3])),
        )
        x_cand = self.vector_from_values(values)
        # Get the cost and constraint matrices
        var_dict = self.get_variable_dict(use_cached=False)
        C = self.get_sdp_cost(var_dict)
        As, bs = self.get_sdp_constraints(var_dict)
        # Optimal cost
        rho = (x_cand.T @ C @ x_cand).item()
        # verbose option
        if verbose:
            self.cert_params.verbose = True
        # Run certifier
        certifier = AnalyticCenter(C, rho, As, bs, self.cert_params)
        result = certifier.certify(x_cand)
        return result


def create_stereo_localization_problem(N_map:int=50, device="cpu", seed=0):
    set_seed(seed)
    torch_device = torch.device(device)
    batch_size = 1
    r_v0s, C_v0s, r_ls = get_gt_setup(
        N_map=N_map, N_batch=batch_size, traj_type="circle", n_turns=0.25
    )
    r_v0s = torch.tensor(r_v0s, device=torch_device)
    C_v0s = torch.tensor(C_v0s, device=torch_device)
    r_ls = torch.tensor(r_ls, device=torch_device)[None, :, :].expand(
        batch_size, -1, -1
    )

    stereo_cam = StereoCameraModel(0.0, 0.0, 484.5, 0.24).to(torch_device)

    pert = 0.0
    xi_pert = torch.tensor([[pert, pert, pert, pert, pert, pert]], device=torch_device)
    T_s_v = se3_exp(xi_pert)[0]

    cam_coords_v = torch.bmm(C_v0s, r_ls - r_v0s)
    cam_coords_v = torch.concat(
        [cam_coords_v, torch.ones(batch_size, 1, r_ls.size(2), device=torch_device)],
        dim=1,
    )

    src_coords_v = torch.concat(
        [r_ls, torch.ones(batch_size, 1, r_ls.size(2), device=torch_device)], dim=1
    )

    cam_coords = T_s_v[None, :, :].bmm(cam_coords_v)
    src_coords = T_s_v[None, :, :].bmm(src_coords_v)

    zeros = torch.zeros(batch_size, 1, 3, device=torch_device).type_as(r_v0s)
    one = torch.ones(batch_size, 1, 1, device=torch_device).type_as(r_v0s)
    r_0v_v = -C_v0s.bmm(r_v0s)
    trans_cols = torch.cat([r_0v_v, one], dim=1)
    rot_cols = torch.cat([C_v0s, zeros], dim=1)
    T_trg_src = torch.cat([rot_cols, trans_cols], dim=2)

    weights = torch.ones(batch_size, 1, src_coords.size(2), device=torch_device)

    valid = weights > 0
    inv_cov_weights, _ = get_inv_cov_weights(cam_coords, valid, stereo_cam)
    # convert to list
    inv_cov_weights = [
        inv_cov_weights[0, i, :, :].cpu().numpy()
        for i in range(inv_cov_weights.size(1))
    ]

    stereo_loc = SinglePoseStereoLocalization(
        keypoints_3D_src=src_coords[0, :3, :].cpu().numpy(),
        keypoints_3D_trg=cam_coords[0, :3, :].cpu().numpy(),
        weights=weights[0, 0, :].cpu().numpy(),
        inv_cov_weights=inv_cov_weights,
        T_s_v=T_s_v,
    )
    # Store actual pose
    stereo_loc.T_trg_src = T_trg_src[0].cpu().numpy()
    return stereo_loc


if __name__ == "__main__":
    stereo_loc = create_stereo_localization_problem()
