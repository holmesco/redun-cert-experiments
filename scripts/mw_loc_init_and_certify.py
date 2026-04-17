import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_targ_frames_3d(
    stereo_loc, T_ests, is_global_min, T_inits=None, axis_scale=0.2, title=None
):
    """Plot target frames in 3D from estimated transforms.

    Args:
        stereo_loc: StereoLocalizationProblem instance.
        T_ests (torch.Tensor): Batched transforms of shape (N, 4, 4).
        is_global_min (array-like): Boolean mask of shape (N,) where True indicates global minima.
        T_inits (torch.Tensor | None): Optional batched initialization transforms of shape (N, 4, 4).
        axis_scale (float): Length of frame axes used in the visualization.
        title (str | None): Optional figure title.
    """
    if T_ests.ndim != 3 or T_ests.shape[1:] != (4, 4):
        raise ValueError("T_ests must have shape (N, 4, 4).")

    is_global_min = np.asarray(is_global_min, dtype=bool)
    if is_global_min.shape[0] != T_ests.shape[0]:
        raise ValueError("is_global_min must have shape (N,) matching T_ests.")

    if T_inits is not None:
        if T_inits.ndim != 3 or T_inits.shape[1:] != (4, 4):
            raise ValueError("T_inits must have shape (N, 4, 4).")
        if T_inits.shape[0] != T_ests.shape[0]:
            raise ValueError("T_inits must have the same batch size as T_ests.")

    # Plots should use the inverse transform T_source_target
    T_ests_plot = torch.linalg.inv(T_ests)
    T_ests_np = T_ests_plot.detach().cpu().numpy()

    if T_inits is not None:
        T_inits_plot = torch.linalg.inv(T_inits)
        T_inits_np = T_inits_plot.detach().cpu().numpy()
    else:
        T_inits_np = None

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    init_global_color = "tab:green"
    init_local_color = "tab:red"
    opt_global_color = "black"
    opt_local_color = "tab:purple"

    # Plot initializations with transparency and color by convergence result
    if T_inits_np is not None:
        for i in range(T_inits_np.shape[0]):
            T0 = T_inits_np[i]
            R0 = T0[:3, :3]
            t0 = T0[:3, 3]
            c0 = init_global_color if is_global_min[i] else init_local_color

            ax.scatter(t0[0], t0[1], t0[2], c=c0, s=35, alpha=0.5)

            ex0 = R0[:, 0] * axis_scale
            ey0 = R0[:, 1] * axis_scale
            ez0 = R0[:, 2] * axis_scale

            ax.quiver(
                t0[0],
                t0[1],
                t0[2],
                ex0[0],
                ex0[1],
                ex0[2],
                color=c0,
                linewidth=1.0,
                alpha=0.5,
            )
            ax.quiver(
                t0[0],
                t0[1],
                t0[2],
                ey0[0],
                ey0[1],
                ey0[2],
                color=c0,
                linewidth=1.0,
                alpha=0.4,
            )
            ax.quiver(
                t0[0],
                t0[1],
                t0[2],
                ez0[0],
                ez0[1],
                ez0[2],
                color=c0,
                linewidth=1.0,
                alpha=0.3,
            )

    # Plot optimized poses: global in black, local in purple
    for i in range(T_ests_np.shape[0]):
        T = T_ests_np[i]
        R = T[:3, :3]
        t = T[:3, 3]
        color = opt_global_color if is_global_min[i] else opt_local_color

        ax.scatter(t[0], t[1], t[2], c=color, s=40)

        ex = R[:, 0] * axis_scale
        ey = R[:, 1] * axis_scale
        ez = R[:, 2] * axis_scale

        ax.quiver(t[0], t[1], t[2], ex[0], ex[1], ex[2], color=color, linewidth=1.0)
        ax.quiver(
            t[0],
            t[1],
            t[2],
            ey[0],
            ey[1],
            ey[2],
            color=color,
            linewidth=1.0,
            alpha=0.75,
        )
        ax.quiver(
            t[0],
            t[1],
            t[2],
            ez[0],
            ez[1],
            ez[2],
            color=color,
            linewidth=1.0,
            alpha=0.5,
        )

    # Plot source keypoints (3D map points)
    if (
        stereo_loc.keypoints_3D_src.ndim == 3
        and stereo_loc.keypoints_3D_src.shape[1] >= 3
    ):
        keypoints_src = stereo_loc.keypoints_3D_src[0, :3, :].detach().cpu().numpy().T
        ax.scatter(
            keypoints_src[:, 0],
            keypoints_src[:, 1],
            keypoints_src[:, 2],
            c="tab:blue",
            s=12,
            alpha=0.6,
            label="keypoints_3D_src",
        )

    if T_ests_np.shape[0] > 0:
        xyz = T_ests_np[:, :3, 3]
        if T_inits_np is not None and T_inits_np.shape[0] > 0:
            xyz = np.vstack([xyz, T_inits_np[:, :3, 3]])
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        pad = 0.15 * span
        ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
        ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
        ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title if title is not None else "Estimated target frames")
    ax.set_aspect("equal", adjustable="box")

    ax.scatter([], [], [], c=init_global_color, s=35, alpha=0.5, label="init → global")
    ax.scatter([], [], [], c=init_local_color, s=35, alpha=0.5, label="init → local")
    ax.scatter([], [], [], c=opt_global_color, s=40, label="optimized global")
    ax.scatter([], [], [], c=opt_local_color, s=40, label="optimized local")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def run_inits_and_certify(
    stereo_loc, N_init=10, seed=0, plot_results=False, plot_axis_scale=0.2
):
    """Generate random initializations, run local optimization, and certify each result.

    Args:
        stereo_loc: StereoLocalizationProblem instance.
        N_init (int): Number of random initializations.
        seed (int): Random seed for reproducibility.
        plot_results (bool): If True, plot costs and estimated frames.
        plot_axis_scale (float): Axis length for plotted frame triads.

    Returns:
        pd.DataFrame: Per-run results including cost, certification status, and timing.
    """
    if stereo_loc.batch_size != 1:
        raise ValueError("run_inits_and_certify currently supports batch_size=1.")
    if stereo_loc.T_trg_src is None:
        raise ValueError(
            "T_trg_src is required for this analysis. "
            "Use create_stereo_localization_problem(...)."
        )

    radius = torch.linalg.norm(stereo_loc.T_trg_src[0, :3, 3]).item()
    r_v0s_init, C_v0s_init = stereo_loc.get_random_inits(
        radius=radius,
        N_batch=N_init,
        seed=seed,
    )
    r_v0s_init = torch.tensor(
        r_v0s_init, dtype=stereo_loc.T_trg_src.dtype, device=stereo_loc.T_trg_src.device
    )
    C_v0s_init = torch.tensor(
        C_v0s_init, dtype=stereo_loc.T_trg_src.dtype, device=stereo_loc.T_trg_src.device
    )

    zeros = torch.zeros(
        N_init,
        1,
        3,
        dtype=stereo_loc.T_trg_src.dtype,
        device=stereo_loc.T_trg_src.device,
    )
    one = torch.ones(
        N_init,
        1,
        1,
        dtype=stereo_loc.T_trg_src.dtype,
        device=stereo_loc.T_trg_src.device,
    )
    r_0v_v = -C_v0s_init.bmm(r_v0s_init)
    trans_cols = torch.cat([r_0v_v, one], dim=1)
    rot_cols = torch.cat([C_v0s_init, zeros], dim=1)
    T_inits = torch.cat([rot_cols, trans_cols], dim=2)

    t_sdp_start = time.perf_counter()
    _, info_sdp, _ = stereo_loc.certifier.solve_sdp(verbose=False)
    t_sdp_end = time.perf_counter()
    sdp_wall_time_s = t_sdp_end - t_sdp_start
    sdp_solver_time_s = info_sdp[0].get("time", np.nan) if len(info_sdp) > 0 else np.nan

    C = stereo_loc.certifier.Cs[0].detach().cpu().numpy()
    results = []
    T_est_list = []

    for i in range(N_init):
        T_init = T_inits[i : i + 1].to(stereo_loc.device)
        T_est, runtime_gtsam = stereo_loc.solve_factor_graph(
            T_init[0].cpu().numpy(), verbose=False
        )
        T_est = torch.from_numpy(T_est[None, :, :])
        T_est_list.append(T_est.detach())
        x_cand = stereo_loc.certifier.transform_to_x(T_est)[0].detach().cpu().numpy()

        optimal_cost = float((x_cand.T @ C @ x_cand).item())
        cert_result = stereo_loc.certifier.certify_solution(
            x_cand,
            verbose=False,
            cost=optimal_cost,
        )

        results.append(
            {
                "init": i,
                "optimal_cost": optimal_cost,
                "certified": bool(cert_result.certified),
                "min_eig": float(cert_result.min_eig),
                "complementarity": float(cert_result.complementarity),
                "certifier_time_s": float(cert_result.solver_time),
                "gtsam_time_s": float(runtime_gtsam),
            }
        )

    T_ests = torch.cat(T_est_list, dim=0)

    df = pd.DataFrame(results)
    print(df)
    print(f"SDP solve wall time: {sdp_wall_time_s:.6f} s")
    print(f"SDP reported solver time: {sdp_solver_time_s:.6f} s")

    global_mask = df["certified"]
    local_mask = ~global_mask

    global_avg_gtsam_time = (
        df.loc[global_mask, "gtsam_time_s"].mean() if global_mask.any() else np.nan
    )
    local_avg_gtsam_time = (
        df.loc[local_mask, "gtsam_time_s"].mean() if local_mask.any() else np.nan
    )
    print(
        "Average local factor graph solve time "
        f"(global minima): {global_avg_gtsam_time:.6f} s"
    )
    print(
        "Average local factor graph solve time "
        f"(local minima): {local_avg_gtsam_time:.6f} s"
    )

    global_avg_cert_time = (
        df.loc[global_mask, "certifier_time_s"].mean() if global_mask.any() else np.nan
    )
    local_avg_cert_time = (
        df.loc[local_mask, "certifier_time_s"].mean() if local_mask.any() else np.nan
    )

    print(f"Average certifier time (global minima): {global_avg_cert_time:.6f} s")
    print(f"Average certifier time (local minima): {local_avg_cert_time:.6f} s")

    global_costs = df.loc[global_mask, "optimal_cost"]
    local_costs = df.loc[local_mask, "optimal_cost"]

    global_cost_mean = global_costs.mean() if global_mask.any() else np.nan
    global_cost_std = global_costs.std(ddof=0) if global_mask.any() else np.nan
    print(f"Global minima cost mean: {global_cost_mean:.6e}")
    print(f"Global minima cost std: {global_cost_std:.6e}")

    global_cost_threshold = global_cost_mean + global_cost_std
    if local_mask.any() and global_mask.any():
        all_local_greater = bool((local_costs > global_cost_threshold).all())
        print(
            "All local minima costs > "
            f"(global mean + std = {global_cost_threshold:.6e}): {all_local_greater}"
        )
    elif local_mask.any() and not global_mask.any():
        print(
            "Cannot compare local minima costs to global mean+std (no global minima found)."
        )
    else:
        print("No local minima found to compare against global mean+std threshold.")

    if plot_results:
        fig_cost, ax_cost = plt.subplots(figsize=(9, 4))
        idx = df["init"].to_numpy()
        ax_cost.scatter(
            idx[global_mask.to_numpy()],
            global_costs.to_numpy(),
            c="black",
            s=35,
            label="global minima cost",
        )
        if local_mask.any():
            ax_cost.scatter(
                idx[local_mask.to_numpy()],
                local_costs.to_numpy(),
                c="tab:purple",
                s=35,
                label="local minima cost",
            )

        if global_mask.any():
            ax_cost.axhline(
                global_cost_mean,
                color="tab:green",
                linestyle="-",
                linewidth=1.5,
                label="global mean",
            )
            ax_cost.axhline(
                global_cost_mean + global_cost_std,
                color="tab:green",
                linestyle="--",
                linewidth=1.0,
                label="global mean ± std",
            )
            ax_cost.axhline(
                global_cost_mean - global_cost_std,
                color="tab:green",
                linestyle="--",
                linewidth=1.0,
            )

        ax_cost.set_xlabel("initialization index")
        ax_cost.set_ylabel("optimal cost")
        ax_cost.set_title("Cost by trial with global mean/std")
        ax_cost.legend(loc="best")
        ax_cost.grid(alpha=0.25)
        plt.tight_layout()

        is_global_min = df["certified"].to_numpy(dtype=bool)
        plot_targ_frames_3d(
            stereo_loc,
            T_ests,
            is_global_min,
            T_inits=T_inits,
            axis_scale=plot_axis_scale,
            title="Initializations and optimized target frames",
        )

    return df


if __name__ == "__main__":
    from mat_weight_loc.stereo_loc import create_stereo_localization_problem

    stereo_loc = create_stereo_localization_problem(batch_size=1, N_map=50)
    # stereo_loc.certifier.check_constraint_linear_independence()
    df = run_inits_and_certify(stereo_loc, N_init=100, plot_results=True, seed=0)
