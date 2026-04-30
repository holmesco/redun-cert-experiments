import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import gtsam

from mwcerts.cert_factor_graph import LocalizationFactorGraph


@dataclass
class PGOProblem:
    factor_graph: LocalizationFactorGraph
    pose_ids: list[gtsam.Symbol]
    ground_truth_poses: list[gtsam.Pose3]
    seed: int
    n_poses: int


def relative_pose_case(n_poses=5, seed: int | None = 7):
    """Tests relative pose measurements and prior pose with no landmarks."""
    if seed is not None:
        np.random.seed(seed)

    # Build factor graph.
    fg = LocalizationFactorGraph()
    pose_ids = []
    poses_gt = []
    for i in range(n_poses):
        # ids
        pose_id = gtsam.Symbol("x", i)
        pose_ids.append(pose_id)
        # Ground-truth pose.
        rot_gt = gtsam.Rot3.RzRyRx(*np.random.randn(3))
        trans_gt = gtsam.Point3(*np.random.randn(3))
        poses_gt.append(gtsam.Pose3(rot_gt, trans_gt))

        if i == 0:
            # Add a prior on the first pose to fix gauge freedom.
            fg.add_prior_pose_factor(
                pose_id=pose_id,
                pose_meas=poses_gt[i],
                weight_rot=1.0,
                weight_trans=1.0,
            )
        else:
            # Add a between factor between this pose and the previous pose.
            fg.add_between_factor(
                pose_id_i=pose_ids[i],
                pose_id_j=pose_ids[i - 1],
                relative_pose=poses_gt[i].compose(poses_gt[i - 1].inverse()),
                weight_rot=1.0,
                weight_trans=1.0,
            )

    # Add constraints
    fg.add_constraints()

    # Adjust cost offset for certification.
    fg.cert_params.eps_cost = 5e-3 / np.trace(fg.get_sdp_cost())
    fg.cert_params.lin_solve_tol = 1e-6
    fg.cert_params.max_iter = 50

    return PGOProblem(
        factor_graph=fg,
        pose_ids=pose_ids,
        ground_truth_poses=poses_gt,
        seed=seed,
        n_poses=n_poses,
    )


def _make_initial_estimate(
    problem: PGOProblem,
    seed: int | None = None,
    pose_rot_sigma: float = 0.05,
    pose_trans_sigma: float = 0.05,
) -> gtsam.Values:
    """Create a perturbed initial estimate from the ground-truth poses."""
    rng = np.random.default_rng(seed)
    initial = gtsam.Values()

    for pose_id, pose_gt in zip(problem.pose_ids, problem.ground_truth_poses):
        rot_noise = gtsam.Rot3.RzRyRx(*(pose_rot_sigma * rng.standard_normal(3)))
        trans_noise = gtsam.Point3(*(pose_trans_sigma * rng.standard_normal(3)))
        delta_pose = gtsam.Pose3(rot_noise, trans_noise)
        initial.insert(pose_id.key(), pose_gt.compose(delta_pose))

    return initial

def run_optimization(problem: PGOProblem, verbose: bool = True, optimize_max_iterations: int = 100, initial_seed: int = 0):
    # Problem statistics.
    var_dict = problem.factor_graph.get_variable_dict(use_cached=False)
    sdp_cost = problem.factor_graph.get_sdp_cost(var_dict)
    sdp_constraints, _ = problem.factor_graph.get_sdp_constraints(var_dict)

    initial_estimate = _make_initial_estimate(problem, seed=initial_seed)

    optimized_values, local_runtime_s = problem.factor_graph.optimize_graph(
        initial_estimate=initial_estimate,
        max_iterations=optimize_max_iterations,
        verbose=verbose,
    )

    X, info_sdp = problem.factor_graph.solve_sdp(
        verbose=verbose,
        adjust_cost=False,
    )
    sdp_runtime_s = info_sdp["time"]
    
    cert_result = problem.factor_graph.certify_solution(
        optimized_values,
        verbose=verbose,
        adjust_cost=True,
    )
    cert_runtime_s = cert_result.solver_time
    
    # Compute optimal costs
    C = problem.factor_graph.get_sdp_cost(var_dict)
    x_opt = problem.factor_graph.vector_from_values(optimized_values)
    x_sdp = X[:,[0]]
    cost_opt = (x_opt.T @ C @ x_opt).item()
    cost_sdp = (x_sdp.T @ C @ x_sdp).item()
    
    return{
            "num_constraints": len(sdp_constraints),
            "n_poses": int(problem.n_poses),
            "sdp_variable_dim": int(sdp_cost.shape[0]),
            "cost_opt": cost_opt,
            "cost_sdp": cost_sdp,
            "certified": cert_result.certified,
            "sdp_runtime_s": sdp_runtime_s,
            "cert_runtime_s": cert_runtime_s,
            "local_runtime_s": local_runtime_s,            
        }


def run_timing_analysis(
    min_poses: int = 1,
    max_poses: int = 100,
    num_trials: int = 10,
    problems_per_trial: int = 5,
    seed: int | None = 7,
    optimize_max_iterations: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run timing analysis for optimization, SDP solve, and certification.

    Parameters
    ----------
    min_poses, max_poses:
        Endpoints for the logarithmically spaced pose counts.
    num_trials:
        Number of pose-count values to evaluate.
    problems_per_trial:
        Number of random problem instances to generate for each pose count.
    seed:
        Seed for reproducible problem generation.
    optimize_max_iterations:
        Maximum number of Levenberg-Marquardt iterations.
    verbose:
        Forward verbosity to the solver routines.

    Returns
    -------
    pd.DataFrame
        One row per problem instance containing runtimes and problem statistics.
    """
    if min_poses < 1:
        raise ValueError("min_poses must be at least 1.")
    if max_poses < min_poses:
        raise ValueError("max_poses must be greater than or equal to min_poses.")
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1.")
    if problems_per_trial < 1:
        raise ValueError("problems_per_trial must be at least 1.")

    trial_pose_counts = np.rint(
        np.logspace(np.log10(min_poses), np.log10(max_poses), num_trials)
    ).astype(int)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for trial_index, n_poses in enumerate(trial_pose_counts):
        for problem_index in range(problems_per_trial):
            problem_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            initial_seed = int(rng.integers(0, np.iinfo(np.int32).max))

            problem = relative_pose_case(n_poses=int(n_poses), seed=problem_seed)
            result = run_optimization(
                problem=problem,
                verbose=verbose,
                optimize_max_iterations=optimize_max_iterations,
                initial_seed=initial_seed,
            )
            result["trial_index"] = trial_index
            result["problem_index"] = problem_index
            result["problem_seed"] = problem_seed
            
            rows.append(result)
 
    return pd.DataFrame(rows)

if __name__ == "__main__":
    problem = relative_pose_case(n_poses=20, seed=7)
    result = run_optimization(problem, verbose=True, optimize_max_iterations=100)
    print(result)
    
    # df = run_timing_analysis(
    #     min_poses=1,
    #     max_poses=100,
    #     num_trials=10,
    #     problems_per_trial=1,
    #     seed=7,
    #     optimize_max_iterations=100,
    #     verbose=False,
    # )
    # print(df)
