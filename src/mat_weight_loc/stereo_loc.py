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


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


# Default dtype
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)


class MatWeightLocResidual:
    """Stack landmark errors for a single Pose3 variable."""

    def __init__(self, keypoint_src, keypoint_trg: list[np.ndarray]):
        self.keypoint_src = keypoint_src
        self.keypoint_trg = keypoint_trg

    def __call__(
        self,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: list[np.ndarray] | None,
    ) -> np.ndarray:
        pose = values.atPose3(this.keys()[0])
        residual = np.zeros(3)

        if jacobians is not None:
            jacobians[0] = np.zeros((3, 6), order="F")

        if jacobians is not None:
            H_pose = np.zeros((3, 6), order="F")
            H_point = np.zeros((3, 3), order="F")
            keypoint_trg_pred = pose.transformFrom(self.keypoint_src, H_pose, H_point)
            jacobians[0] = H_pose
        else:
            keypoint_trg_pred = pose.transformFrom(self.keypoint_src)

        residual = keypoint_trg_pred - self.keypoint_trg

        return residual


def build_stereo_loc_fg(keypoint_3D_src, keypoint_3D_trg, weight, inv_cov_weight=None):

    n_points = keypoint_3D_src.shape[1]
    # create expression leaf for pose variable
    T_trg_src_key = X(0)
    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    # Loop through points and add factors
    for i in range(n_points):
        # Extract the i-th keypoint from source and target
        src_point = keypoint_3D_src[:3, i]
        trg_point = keypoint_3D_trg[:3, i]

        # Get the noise model for this landmark
        if inv_cov_weight is not None:
            # Use the inverse covariance weight to create a noise model
            noise_model = gtsam.noiseModel.Gaussian.Information(
                inv_cov_weight[i] * weight[i]
            )
        else:
            # Use the weight to create a noise model (assuming isotropic noise)
            noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0 / weight)

        # Add a factor between the source and target keypoints using the pose variable
        factor = gtsam.CustomFactor(
            noise_model, [T_trg_src_key], MatWeightLocResidual(src_point, trg_point)
        )
        graph.add(factor)
    return graph


def solve_stereo_loc_fg(graph, T_init, verbose=False):
    # Create initial values
    values = gtsam.Values()
    values.insert(X(0), gtsam.Pose3(T_init))
    # Create optimizer
    opt_params = gtsam.LevenbergMarquardtParams()
    if verbose:
        opt_params.setVerbosityLM("SUMMARY")
    else:
        opt_params.setVerbosityLM("SILENT")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, opt_params)
    # Optimize
    start_time = time.perf_counter()
    result = optimizer.optimize()
    runtime_s = time.perf_counter() - start_time
    optimized_pose = result.atPose3(X(0)).matrix()
    if verbose:
        print(f"Optimization runtime: {runtime_s:.6f} s")
        print("Optimization result:\n", optimized_pose)

    return optimized_pose, runtime_s


class StereoLocalizationProblem:
    def __init__(self, keypoints_3D_src, keypoints_3D_trg, weights, stereo_cam, T_s_v):

        self.keypoints_3D_src = keypoints_3D_src
        self.keypoints_3D_trg = keypoints_3D_trg
        self.weights = weights
        self.T_s_v = T_s_v
        self.stereo_cam = stereo_cam

        self.batch_size = self.keypoints_3D_src.size(0)
        self.N_map = self.keypoints_3D_src.size(2)
        self.device = self.keypoints_3D_src.device
        self.T_trg_src = None

        # Get inverse covariance weights
        # Get matrix weights - assuming 0.5 pixel std dev
        valid = self.weights > 0
        self.inv_cov_weights, cov = get_inv_cov_weights(
            self.keypoints_3D_trg, valid, self.stereo_cam
        )
        # Intialize certifier class
        self.certifier = StereoPoseCertifier(
            self.T_s_v,
            self.keypoints_3D_src,
            self.keypoints_3D_trg,
            self.weights,
            self.inv_cov_weights,
        )
        # Build factor graph
        self.factor_graph = build_stereo_loc_fg(
            self.keypoints_3D_src[0].cpu().numpy(),
            self.keypoints_3D_trg[0].cpu().numpy(),
            self.weights[0][0].cpu().numpy(),
            self.inv_cov_weights[0].cpu().numpy(),
        )

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

    def solve_factor_graph(self, T_init, verbose=False):
        T_est, runtime = solve_stereo_loc_fg(self.factor_graph, T_init, verbose=verbose)

        return T_est, runtime


def create_stereo_localization_problem(batch_size=1, N_map=50, device="cuda:0", seed=0):
    set_seed(seed)
    torch_device = torch.device(device)

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

    stereo_loc = StereoLocalizationProblem(
        keypoints_3D_src=src_coords,
        keypoints_3D_trg=cam_coords,
        weights=weights,
        stereo_cam=stereo_cam,
        T_s_v=T_s_v,
    )
    stereo_loc.T_trg_src = T_trg_src
    return stereo_loc
