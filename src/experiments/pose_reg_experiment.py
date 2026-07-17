"""Pose registration certification experiment.

This experiment stresses the certifier of the
:class:`PointCloudRegistrationBlock` on synthetic pose registration problems
generated from the Stanford bunny point cloud.

Setup (per problem instance):

* Sample ``N`` points from the (rescaled, zero-centered) bunny model. These are
  the *source* keypoints, expressed in the model/world frame.
* Build a "camera frame" whose z axis points at the centroid of the cloud and
  whose origin is offset from the centroid by a parameterized distance. A
  slight y-axis rotation (parameterized by ``theta``) is applied to the camera
  pose to obtain some parallax. The relative transform to the camera frame is
  the ground-truth registration solution.
* The *target* keypoints are the source points expressed in the camera frame,
  corrupted by noise drawn from the linearized stereo camera covariance (the
  same model used by ``get_inv_cov_weights``).

Each trial then mirrors the pose registration portion of
:meth:`StereoPipeline.forward`: matrix weights are computed with
``get_inv_cov_weights``, the pose is estimated with the factor graph solver
from an initial value that is either the ground truth or a random perturbation
of it (see :class:`InitType`), and the solution is certified.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import open3d as o3d

from stereo_loc.AnalyticCenterParamsConfig import AnalyticCenterParamsConfig
from stereo_loc.PointCloudRegistrationBlock import (
    PointCloudRegistrationBlock,
    PointCloudRegistrationConfig,
)
from utils.keypoint_tools import get_inv_cov_weights
from utils.stereo_camera_model import StereoCameraModel, StereoCameraConfig

ROOT = Path(__file__).resolve().parents[2]


class InitType(Enum):
    """How the initial value for the factor graph solver is generated."""

    # Initialize at the ground-truth transform.
    GROUNDTRUTH = "GROUNDTRUTH"
    # Initialize at a random perturbation of the ground-truth transform (see
    # ``init_rot_pert_max`` / ``init_trans_pert_max``).
    RANDOM = "RANDOM"


@dataclass
class PoseRegExperimentConfig:
    # Name of the experiment, used for saving results.
    experiment_name: str = "pose_reg_default"
    # Path (relative to ROOT) to the .ply point cloud used to generate problems.
    ply_path: Path = Path("data/bun10k.ply")
    # Side length of the cube the point cloud is rescaled into.
    scale_cube_size: float = 1.0
    # Seed for reproducibility.
    seed: int = 0
    # Save results
    save_results: bool = True
    # Plot the final problem instance (source cloud + camera frame) at the end.
    plot: bool = False

    # --- Problem setup parameters (sweeps) ---
    # Numbers of points sampled from the bunny model.
    num_points: List[int] = field(default_factory=lambda: [50])
    # Offsets of the camera frame from the cloud centroid, along the optical axis.
    camera_distances: List[float] = field(default_factory=lambda: [3.0])
    # y-axis rotations (radians) applied to the camera pose to obtain parallax.
    thetas: List[float] = field(default_factory=lambda: [0.2])

    # --- Stereo camera model used for noise generation and matrix weights ---
    stereo_camera_config: StereoCameraConfig = field(
        default_factory=lambda: StereoCameraConfig(
            cu=0.0, cv=0.0, f=484.5, b=0.24, sigma=0.5
        )
    )
    # If True, corrupt the target keypoints with noise drawn from the linearized
    # stereo camera covariance.
    add_noise: bool = True

    # --- Registration / certification config ---
    registration_config: PointCloudRegistrationConfig = field(
        default_factory=lambda: PointCloudRegistrationConfig(
            certify=True,
            ac_params=AnalyticCenterParamsConfig(verbose=False),
        )
    )
    # Number of reference SDP solves per problem instance; cost and solve time
    # are averaged over these.
    n_sdp_trials: int = 5
    # Number of warmup SDP solves per problem instance; these are skipped when
    # computing the average cost and solve time to avoid any warm-start bias.
    n_sdp_warmup: int = 3

    # --- Trial parameters (initial values) ---
    # How the initial value is generated for each trial.
    init_type: InitType = InitType.RANDOM
    # Number of trials (different initial values) per problem instance.
    num_trials: int = 20
    # Maximum rotation perturbation (radians) for random initial values.
    init_rot_pert_max: float = 3.14159265
    # Maximum translation perturbation (norm) for random initial values.
    init_trans_pert_max: float = 1.0
    # Relative cost tolerance for flagging convergence to the global optimum.
    global_cost_rtol: float = 1e-4


def load_experiment_config(config_path: Path) -> PoseRegExperimentConfig:
    # Start with defaults from the dataclass.
    config = OmegaConf.structured(PoseRegExperimentConfig)
    # Merge overrides if provided.
    if config_path:
        overrides = OmegaConf.load(ROOT / config_path)
        config = OmegaConf.merge(config, overrides)
    return OmegaConf.to_object(config)


# ----------------------------------------------------------------------------
# Point cloud / problem setup helpers
# ----------------------------------------------------------------------------


def read_ply(ply_path: Path) -> np.ndarray:
    """Read x-y-z vertices from a PLY file into an (N, 3) array."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        raise ValueError(f"No points found in {ply_path}")

    return pts


def scale_to_cube(pts: np.ndarray, s: float) -> np.ndarray:
    """Rescale a point cloud so that its largest extent equals ``s``."""
    extent = pts.max(axis=0) - pts.min(axis=0)
    sf = extent.max()
    return pts * (s / sf)


def get_camera_transform(distance: float, theta: float) -> np.ndarray:
    """Build the camera pose ``T_w_c`` (camera frame expressed in the world frame).

    The camera z axis points at the centroid of the cloud (the world origin,
    since the cloud is zero-centered) from a distance ``distance`` along the
    optical axis. ``theta`` rotates the camera pose about the world y axis so
    that the viewpoint is slightly off the -z axis, providing parallax.

    Returns the (4, 4) homogeneous transform ``T_w_c`` such that
    ``p_w = T_w_c @ p_c``.
    """
    # Camera axes in the world frame.
    R_w_c = Rotation.from_euler("y", theta).as_matrix()
    # Place the camera so its z axis points at the centroid from ``distance``:
    # the camera center plus ``distance`` times the optical axis is the origin.
    z_axis_w = np.eye(3)[:, 2]
    camera_center_w = -distance * z_axis_w
    T_w_c = np.eye(4)
    T_w_c[:3, :3] = R_w_c
    T_w_c[:3, 3] = camera_center_w
    return T_w_c


def pose_reg_setup(
    cfg: PoseRegExperimentConfig,
    pcd0: np.ndarray,
    n: int,
    distance: float,
    theta: float,
    stereo_cam: StereoCameraModel,
    rng: np.random.Generator,
):
    """Generate a single pose registration problem instance.

    Samples ``n`` points from the bunny, expresses them in the camera frame
    (with stereo-camera noise if enabled) and computes the matrix weights via
    ``get_inv_cov_weights``.

    Returns ``(kpt_3D_src, kpt_3D_trg, inv_cov_weights, T_w_c)`` where the
    keypoints are homogeneous (4, n) tensors, the weights have shape
    (n, 3, 3) and ``T_w_c`` is the ground-truth camera pose. The ground-truth
    registration solution (``T_src_trg``) is ``T_w_c`` itself, since the source
    keypoints live in the world frame and the target keypoints in the camera
    frame.
    """
    # Set default datatype to float32 for PyTorch tensors.
    torch.set_default_dtype(torch.float32)
    # Sample n points from the bunny model (world/source frame).
    idx = rng.choice(pcd0.shape[0], size=n, replace=False)
    pts_w = pcd0[idx]  # (n, 3)
    kpt_3D_src = torch.ones(4, n, dtype=torch.float32)
    kpt_3D_src[:3, :] = torch.from_numpy(pts_w.T).float()

    # Ground-truth camera pose and target keypoints in the camera frame.
    T_w_c = get_camera_transform(distance, theta)
    T_c_w = np.linalg.inv(T_w_c)
    kpt_3D_trg = torch.from_numpy(T_c_w).float() @ kpt_3D_src

    # Corrupt the target keypoints with noise drawn from the linearized stereo
    # camera covariance at the (noiseless) camera-frame points.
    valid = torch.ones(1, 1, n, dtype=bool)
    if cfg.add_noise:
        _, cov_cam = get_inv_cov_weights(
            kpt_3D_trg.unsqueeze(0), valid, stereo_cam, normalize_weights=True
        )
        L = torch.linalg.cholesky(cov_cam[0])  # (n, 3, 3)
        noise = torch.from_numpy(rng.standard_normal((n, 3, 1))).float()
        kpt_3D_trg[:3, :] = kpt_3D_trg[:3, :] + L.bmm(noise).squeeze(2).T

    # Matrix weights from the measured (noisy) camera-frame keypoints, matching
    # the pose registration code in StereoPipeline.forward.
    inv_cov_weights, _ = get_inv_cov_weights(
        kpt_3D_trg.unsqueeze(0), valid, stereo_cam, normalize_weights=True
    )

    return kpt_3D_src, kpt_3D_trg, inv_cov_weights.squeeze(0), T_w_c


def sample_initial_pose(
    T_gt: np.ndarray,
    init_type: InitType,
    rot_pert_max: float,
    trans_pert_max: float,
    rng: np.random.Generator,
):
    """Generate an initial value ``T_init`` (``T_src_trg``) for the solver.

    Returns ``(T_init, rot_pert, trans_pert)`` where the perturbation
    magnitudes are zero for ``InitType.GROUNDTRUTH``.
    """
    if init_type == InitType.GROUNDTRUTH:
        return T_gt.copy(), 0.0, 0.0

    # Random rotation perturbation (random axis, angle in [0, rot_pert_max]).
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rot_pert = rng.uniform(0.0, rot_pert_max)
    dR = Rotation.from_rotvec(rot_pert * axis).as_matrix()
    # Random translation perturbation (random direction, norm in [0, trans_pert_max]).
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    trans_pert = rng.uniform(0.0, trans_pert_max)

    T_init = T_gt.copy()
    T_init[:3, :3] = dR @ T_gt[:3, :3]
    T_init[:3, 3] = T_gt[:3, 3] + trans_pert * direction
    return T_init, rot_pert, trans_pert


def sample_initial_pose_on_sphere(
    distance: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample an initial camera pose on a sphere around the point cloud.

    The camera center is uniformly distributed on a sphere of radius
    ``distance`` centered at the cloud centroid (the world origin) and the
    camera z axis points at the centroid, with a random roll about it.

    Returns ``T_init`` (``T_src_trg``, i.e. the camera pose in the world frame).
    """
    # Random location on the sphere.
    center = rng.normal(size=3)
    center = distance * center / np.linalg.norm(center)
    # z axis points at the centroid (origin); y is a random direction
    # orthogonal to z (random roll) and x completes the right-handed frame.
    z = -center / np.linalg.norm(center)
    y = rng.normal(size=3)
    y = y - (y @ z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)

    T_init = np.eye(4)
    T_init[:3, :3] = np.column_stack([x, y, z])
    T_init[:3, 3] = center
    return T_init


def pose_errors(T_est: np.ndarray, T_gt: np.ndarray):
    """Rotation (radians) and translation (norm) errors of ``T_est`` w.r.t. ``T_gt``."""
    T_err = np.linalg.inv(T_est) @ T_gt
    rot_error = np.linalg.norm(Rotation.from_matrix(T_err[:3, :3]).as_rotvec())
    trans_error = np.linalg.norm(T_err[:3, 3])
    return rot_error, trans_error


def plot_experiment(
    kpt_3D_src: torch.Tensor,
    T_w_c: np.ndarray,
    trial_frames: list[tuple[np.ndarray, np.ndarray, bool]] | None = None,
):
    """Plot the source cloud, the true camera frame and the per-trial frames.

    The ground-truth camera frame is drawn with RGB (x, y, z) axes. For each
    trial, the initialization frame ``T_init`` is drawn with alpha 0.3 (green if
    the trial's solution was certified, red otherwise) and the solution frame
    ``T_est`` is drawn fully opaque (magenta if certified, orange otherwise).

    Args:
        kpt_3D_src: homogeneous (4, n) source keypoints in the world frame.
        T_w_c: ground-truth camera pose (4, 4).
        trial_frames: list of ``(T_init, T_est, certified)`` tuples per trial.
            Both transforms are ``T_src_trg`` (== camera pose in world frame).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pts = kpt_3D_src[:3, :].cpu().numpy().T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="magenta", marker=".", label="Source")

    def draw_frame(T, scale, color=None, alpha=1.0, ls="-", lw=1.5):
        """Draw a coordinate frame; RGB axes if ``color`` is None, else one color."""
        origin = T[:3, 3]
        axis_colors = ["red", "green", "blue"] if color is None else [color] * 3
        for axis_idx, c in enumerate(axis_colors):
            tip = origin + scale * T[:3, axis_idx]
            ax.plot(
                [origin[0], tip[0]],
                [origin[1], tip[1]],
                [origin[2], tip[2]],
                color=c,
                alpha=alpha,
                linestyle=ls,
                linewidth=lw,
            )

    # Ground-truth camera frame (RGB axes).
    draw_frame(T_w_c, scale=0.5, color="black")

    # Per-trial initialization and solution frames.
    if trial_frames is not None:
        for T_init, T_est, certified in trial_frames:
            draw_frame(
                T_init,
                scale=0.3,
                color="green" if certified else "red",
                alpha=1.0,
                lw=0.4,
            )
            draw_frame(
                T_est,
                scale=0.3,
                color="magenta" if certified else "orange",
                alpha=1.0,
            )

    # legend_handles = [
    #     Line2D([0], [0], color="green", alpha=0.3, label="Init (certified)"),
    #     Line2D([0], [0], color="red", alpha=0.3, label="Init (not certified)"),
    #     Line2D([0], [0], color="magenta", label="Solution (certified)"),
    #     Line2D([0], [0], color="orange", label="Solution (not certified)"),
    # ]
    # ax.legend(handles=legend_handles)
    # ax.set_title("Pose registration setup (RGB axes: GT camera)")
    # Remove the axes background (panes) and grid.
    ax.grid(False)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.axis("off")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
    # Set camera to good vantage point for the bunny (elev, azim) = (30, -60).
    ax.view_init(elev=54, azim=168, roll=-103)

    # Make the 3D axes fill the whole figure and zoom in so there is no margin.
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    try:
        # `zoom` was added in matplotlib 3.6; fall back gracefully otherwise.
        ax.set_box_aspect(None, zoom=1.6)
    except TypeError:
        pass

    return ax


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    return rng


# ----------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------


def run_experiment(cfg: PoseRegExperimentConfig):
    rng = set_seed(cfg.seed)
    # Load, rescale and zero-center the bunny (centroid at the origin).
    pcd0 = read_ply(ROOT / cfg.ply_path)
    pcd0 = scale_to_cube(pcd0, cfg.scale_cube_size)
    pcd0 -= pcd0.mean(axis=0)

    stereo_cam = StereoCameraModel(cfg.stereo_camera_config)

    output_data = []
    total = (
        len(cfg.num_points)
        * len(cfg.camera_distances)
        * len(cfg.thetas)
        * cfg.num_trials
    )
    index = 0
    with tqdm(total=total, desc="POSE_REG") as pbar:
        for n in cfg.num_points:
            for distance in cfg.camera_distances:
                for theta in cfg.thetas:
                    # Generate the problem instance for this configuration.
                    rng = set_seed(cfg.seed + index)
                    kpt_3D_src, kpt_3D_trg, inv_cov_weights, T_w_c = pose_reg_setup(
                        cfg, pcd0, n, distance, theta, stereo_cam, rng
                    )
                    # Ground-truth registration solution T_src_trg (world <- camera).
                    T_gt = T_w_c

                    # Set up the registration block
                    registration_block = PointCloudRegistrationBlock(
                        config=cfg.registration_config,
                        keypoints_3D_src=kpt_3D_src[:3, :],  # (3, n)
                        keypoints_3D_trg=kpt_3D_trg[:3, :],  # (3, n)
                        inv_cov_weights=inv_cov_weights,  # (n, 3, 3)
                    )

                    # Reference solve from the ground truth to obtain the
                    # (presumed) globally optimal cost for this instance,
                    # averaged over repeated solves.
                    sdp_costs = []
                    sdp_times = []
                    for i in range(cfg.n_sdp_trials + cfg.n_sdp_warmup):
                        T_ref, info_ref = registration_block.solve_sdp(
                            verbose=cfg.registration_config.verbose
                        )
                        if i < cfg.n_sdp_warmup:
                            # Skip the first trials to avoid any warm-start bias.
                            continue
                        sdp_costs.append(info_ref["cost"])
                        sdp_times.append(info_ref["time"])
                    cost_sdp = float(np.mean(sdp_costs))
                    time_sdp = float(np.mean(sdp_times))

                    # Track (T_init, T_est, certified) per trial for plotting.
                    trial_frames = []

                    for trial in range(cfg.num_trials):
                        index += 1
                        # Reset rng for reproducibility across trials.
                        rng = set_seed(cfg.seed + index)

                        # Initial value for this trial: random pose on a sphere
                        # of radius ``distance`` looking at the cloud centroid.
                        T_init = sample_initial_pose_on_sphere(distance, rng)

                        # Solve the factor graph from the initial value.
                        t1 = time.perf_counter()
                        T_est, info = registration_block.solve_factor_graph(
                            T_init, verbose=cfg.registration_config.verbose
                        )
                        t2 = time.perf_counter()
                        t_solver = t2 - t1

                        # Certify the solution.
                        registration_certified = False
                        t_certify = np.nan
                        num_iter_cert = None
                        if cfg.registration_config.certify:
                            cert_result = registration_block.certify_solution(T_est)
                            registration_certified = cert_result.certified
                            t_certify = cert_result.solver_time
                            num_iter_cert = cert_result.num_iterations

                        # Track the initialization and solution frames.
                        trial_frames.append(
                            (T_init.copy(), T_est.copy(), registration_certified)
                        )

                        # Errors w.r.t. the ground truth and the reference cost.
                        rot_error, trans_error = pose_errors(T_est, T_gt)
                        cost = info["cost"]
                        global_min = (cost - cost_sdp) / (
                            1.0 + cost_sdp
                        ) <= cfg.global_cost_rtol

                        output_data.append(
                            dict(
                                num_points=n,
                                camera_distance=distance,
                                theta=theta,
                                trial=trial,
                                cost=cost,
                                cost_ref=cost_sdp,
                                global_min=global_min,
                                cert_reg=registration_certified,
                                t_solver=t_solver,
                                t_sdp=time_sdp,
                                t_certify=t_certify,
                                num_iter_cert=num_iter_cert,
                                rot_error=rot_error,
                                trans_error=trans_error,
                            )
                        )
                        pbar.update(1)

    df = pd.DataFrame(output_data)

    if cfg.save_results:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M")
        run_dir = (
            ROOT / "results" / "pose_registration" / cfg.experiment_name / timestamp
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(run_dir / "results.csv", index=False)
        OmegaConf.save(OmegaConf.structured(cfg), run_dir / "experiment.yaml")
        print(f"\nSaved results to {run_dir}")
    else:
        print("\nExperiment results:")
        print(df)
        print("\nCertification results:")
        print(df[["cost", "global_min", "cert_reg", "num_iter_cert"]])

    # Save the tracked frames for the final problem instance.
    if cfg.save_results and len(trial_frames) > 0:
        np.savez(
            run_dir / "frames.npz",
            T_w_c=T_w_c,
            T_init=np.stack([f[0] for f in trial_frames]),
            T_est=np.stack([f[1] for f in trial_frames]),
            certified=np.array([f[2] for f in trial_frames]),
        )

    # Plot the final problem instance (source cloud + camera frames).
    if cfg.plot:
        # T_src_trg == T_w_c for this problem, so init/solution frames are
        # plotted directly as camera poses in the world frame.
        ax = plot_experiment(kpt_3D_src, T_w_c, trial_frames)
        if cfg.save_results:
            fig_path = run_dir / "setup.png"
            ax.get_figure().savefig(
                fig_path, dpi=1000, bbox_inches="tight", pad_inches=0
            )
            print(f"Saved figure to {fig_path}")
        else:
            from matplotlib import pyplot as plt

            # # Callback function to capture real-time values
            # def on_draw(event):
            #     print(
            #         f"Current View -> Elev: {ax.elev:.1f}°, Azim: {ax.azim:.1f}°, Roll: {ax.roll:.1f}°"
            #     )

            # # Bind the event to the canvas
            # plt.gcf().canvas.mpl_connect("draw_event", on_draw)
            plt.show()

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="pose_reg_test.yaml")
    args = parser.parse_args()

    exp_cfg_path = ROOT / "configs" / "pose_registration_experiments" / args.filename
    exp_config = load_experiment_config(exp_cfg_path)
    run_experiment(exp_config)
