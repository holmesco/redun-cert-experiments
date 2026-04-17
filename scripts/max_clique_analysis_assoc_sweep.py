"""
Max Clique Analysis (Association Sweep)
=======================================
Runs the max clique pipeline (SDP solve + analytic center certification) across
an association-count sweep with a fixed outlier ratio, collecting timing and
problem-size statistics into a pandas DataFrame.

This script will use MOSEK to solve for the IP solution (which is rank 1).

Usage:
    python max_clique_analysis_assoc_sweep.py
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from max_clique.max_clique import generate_dataset, MaxCliqueProblem
from ranktools import AnalyticCenterParams, LinearSolverType, LowRankPrecondMethod

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of association-count values to sweep
N_M_ASSOC = 10

# Number of randomized trials to run per association count
N_TRIALS_PER_M_ASSOC = 10

# Range of association counts (log-spaced between these bounds)
M_ASSOC_MIN = 50
M_ASSOC_MAX = 250

# Fixed outlier ratio for all runs
FIXED_OUTRAT = 0.7

# Number of trials (from the low-association end) for
# which slow solver is also run. Set to N_M_ASSOC
# to run them for every association-count trial.
N_FULL_MAT = 0

# Baseline dataset parameters used to define fixed ratios
BASE_M_ASSOC = 10
BASE_N1 = 10
BASE_N2O = 1

# Fixed sensor/model parameters
SIGMA = 0.01
PCFILE = "/workspace/experiments/data/bun10k.ply"

# Random seed for reproducibility
SEED = 0

# ---------------------------------------------------------------------------
# Solver bookkeeping
# ---------------------------------------------------------------------------

# All solver types to benchmark. MFCG variants are always run; CG and LDLT
# are restricted to the first N_FULL_MAT association-count trials.
ALL_SOLVERS = {
    "MFCG_LRP": LinearSolverType.MFCG_LRP,
    # "MFCG_DP":  LinearSolverType.MFCG_DP,
    # "CG":       LinearSolverType.CG,
    # "LDLT":     LinearSolverType.LDLT,
}

SLOW_SOLVERS = {"MFCG_DP"}


def make_ac_params(solver_type: LinearSolverType) -> AnalyticCenterParams:
    """Return a default AnalyticCenterParams with the given linear solver."""
    params = AnalyticCenterParams()
    params.verbose = True
    params.lin_solver = solver_type
    params.lin_solve_max_iter = 400
    params.lin_solve_tol = 1e-5
    params.lrp_params.tau = 1e-7
    params.delta_init = 1e-6
    params.delta_min = 1e-8
    # turn off rescaling
    params.rescale_lin_sys = False
    # Turn off perturbations:
    params.perturb_constraints = False
    params.perturb_cost = True
    params.adaptive_perturb = True
    params.cost_offset = 1e-4
    # Set preconditioner
    params.lrp_params.method = LowRankPrecondMethod.SparseLDLT

    return params


def scaled_problem_sizes(m_assoc: int) -> tuple[int, int]:
    """Scale N1 and N2O with M_ASSOC while preserving baseline ratios."""
    n1_ratio = BASE_N1 / BASE_M_ASSOC
    n2o_ratio = BASE_N2O / BASE_M_ASSOC

    n1 = max(1, int(round(m_assoc * n1_ratio)))
    n2o = max(1, int(round(m_assoc * n2o_ratio)))
    return n1, n2o


def association_values(
    n_m_assoc: int, m_assoc_min: int, m_assoc_max: int
) -> np.ndarray:
    """Generate unique integer association counts on a log-spaced grid."""
    vals = np.logspace(np.log10(m_assoc_min), np.log10(m_assoc_max), n_m_assoc)
    vals_int = np.unique(np.round(vals).astype(int))
    return vals_int


def run_analysis(
    n_m_assoc: int = N_M_ASSOC,
    m_assoc_min: int = M_ASSOC_MIN,
    m_assoc_max: int = M_ASSOC_MAX,
    fixed_outrat: float = FIXED_OUTRAT,
    n_trials_per_m_assoc: int = N_TRIALS_PER_M_ASSOC,
    n_full_mat: int = N_FULL_MAT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run the max-clique analysis sweep over association count.

    Parameters
    ----------
    n_m_assoc : int
        Number of association-count values to test.
    m_assoc_min, m_assoc_max : int
        Bounds of the log-spaced association-count sweep.
    fixed_outrat : float
        Fixed outlier ratio used for all trials.
    n_trials_per_m_assoc : int
        Number of randomized trials to run for each association count.
    n_full_mat : int
        Number of trials (from the low end) for which expensive solvers
        are also run.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per (m_assoc, solver, trial) combination with columns:
        m_assoc, n1, n2o, outrat, solver, n_constraints,
        sdp_time_s, ac_time_s, certified, min_eig, complementarity,
        sdp_rank, eig_ratio.
    """
    np.random.seed(seed)

    m_assoc_values = association_values(n_m_assoc, m_assoc_min, m_assoc_max)

    # Random ground-truth pose (fixed across trials for consistency)
    T_21 = np.eye(4)
    T_21[:3, :3] = Rotation.random().as_matrix()
    T_21[:3, 3] = np.random.uniform(-5, 5, size=3)

    records: list[dict] = []

    for i_m_assoc, m_assoc in enumerate(m_assoc_values):
        n1, n2o = scaled_problem_sizes(int(m_assoc))

        print(f"\n{'='*70}")
        print(
            f"Associations {i_m_assoc+1}/{len(m_assoc_values)} | "
            f"M_ASSOC={m_assoc}, N1={n1}, N2O={n2o}, outrat={fixed_outrat:.3f}"
        )
        print(f"{'='*70}")

        for i_trial in range(n_trials_per_m_assoc):
            print(f"\n  Trial {i_trial+1}/{n_trials_per_m_assoc}")

            # ---- Generate dataset --------------------------------------------
            clipper, Agt = generate_dataset(
                PCFILE, int(m_assoc), n1, n2o, fixed_outrat, SIGMA, T_21
            )

            # ---- Solve the SDP relaxation ------------------------------------
            # Use MFCG params for the problem setup (solver only matters for AC)
            prob = MaxCliqueProblem(
                clipper,
                params=make_ac_params(LinearSolverType.MFCG_DP),
            )

            X_sdp, u_sdp, sdp_rank, sdp_time, eig_ratio = prob.solve_sdp()

            n_constraints = len(prob.As)
            sdp_cost = -(u_sdp.T @ prob.M @ u_sdp).item()

            # ---- Certify with each requested solver --------------------------
            for solver_name, solver_enum in ALL_SOLVERS.items():
                # Skip expensive full-matrix solvers beyond the first n_full_mat
                if solver_name in SLOW_SOLVERS and i_m_assoc >= n_full_mat:
                    continue

                print(f"\n--- Solver: {solver_name} ---")
                params = make_ac_params(solver_enum)
                prob_solver = MaxCliqueProblem(clipper, params=params)

                result = prob_solver.certify_candidate(u_sdp, cost=sdp_cost)

                records.append(
                    {
                        "m_assoc": int(m_assoc),
                        "n1": n1,
                        "n2o": n2o,
                        "outrat": fixed_outrat,
                        "trial": i_trial,
                        "solver": solver_name,
                        "n_constraints": n_constraints,
                        "sdp_time_s": sdp_time,
                        "sdp_rank": sdp_rank,
                        "ac_time_s": result.solver_time,
                        "certified": result.certified,
                        "min_eig": result.min_eig,
                        "complementarity": result.complementarity,
                        "eig_ratio": eig_ratio,
                    }
                )

    df = pd.DataFrame(records)
    return df


DEFAULT_CSV = "/workspace/experiments/results/max_clique_analysis_assoc_sweep.csv"


def plot_runtime_vs_m_assoc(csv_path: str = DEFAULT_CSV) -> None:
    """Load results CSV and plot runtime vs number of associations."""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(11, 5))

    m_values = np.array(sorted(df["m_assoc"].unique()))
    x = np.arange(len(m_values))

    # --- Interior Point (SDP) ---
    sdp = df.drop_duplicates(subset=["m_assoc", "trial"])[["m_assoc", "sdp_time_s"]]
    series = [
        (
            "Interior Point",
            [sdp[sdp["m_assoc"] == m]["sdp_time_s"].values for m in m_values],
        )
    ]

    for solver_name in ["LDLT", "CG", "MFCG_LRP", "MFCG_DP"]:
        sub = df[df["solver"] == solver_name]
        if sub.empty:
            continue
        series.append(
            (
                solver_name,
                [sub[sub["m_assoc"] == m]["ac_time_s"].values for m in m_values],
            )
        )

    width = 0.8 / max(len(series), 1)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width

    for i, (label, data) in enumerate(series):
        pos = x + offsets[i]
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
        )
        color = f"C{i}"
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for median in bp["medians"]:
            median.set_color(color)
            median.set_linewidth(1.5)
        ax.plot([], [], color=color, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in m_values], rotation=45)
    ax.set_xlabel("Number of associations (M_ASSOC)")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtime by Association Count (Fixed Outlier Ratio)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(csv_path.replace(".csv", "_runtime_box.png"), dpi=150)
    plt.show()


def plot_eig_ratio_vs_m_assoc(csv_path: str = DEFAULT_CSV) -> None:
    """Load results CSV and plot eigenvalue ratio vs association count."""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))

    m_values = np.array(sorted(df["m_assoc"].unique()))
    eig = df.drop_duplicates(subset=["m_assoc", "trial"])[["m_assoc", "eig_ratio"]]
    data = [eig[eig["m_assoc"] == m]["eig_ratio"].values for m in m_values]

    ax.boxplot(data, labels=[str(m) for m in m_values], showfliers=False)
    ax.set_xlabel("Number of associations (M_ASSOC)")
    plt.setp(ax.get_xticklabels(), rotation=45)

    ax.set_ylabel("Eigenvalue ratio")
    ax.set_title("Eigenvalue Ratio by Association Count")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(csv_path.replace(".csv", "_eig_ratio.png"), dpi=150)
    plt.show()


def plot_runtime_scatter_vs_constraints(csv_path: str = DEFAULT_CSV) -> None:
    """Load results CSV and create a scatter plot of runtime vs constraints."""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    # SDP runtime appears once per solver row, so deduplicate by trial instance
    sdp = df.drop_duplicates(subset=["m_assoc", "trial"])
    ax.scatter(
        sdp["n_constraints"],
        sdp["sdp_time_s"],
        marker="s",
        alpha=0.8,
        label="Interior Point",
    )

    for solver_name, marker in [
        ("LDLT", "^"),
        ("CG", "o"),
        ("MFCG_LRP", "D"),
        ("MFCG_DP", "v"),
    ]:
        sub = df[df["solver"] == solver_name]
        if sub.empty:
            continue
        ax.scatter(
            sub["n_constraints"],
            sub["ac_time_s"],
            marker=marker,
            alpha=0.7,
            label=solver_name,
        )

    ax.set_xlabel("Number of constraints")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtime Scatter vs. Number of Constraints")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(csv_path.replace(".csv", "_runtime_scatter.png"), dpi=150)
    plt.show()


def generate_plots(out_path: str) -> None:
    """Generate all analysis plots."""
    plot_runtime_vs_m_assoc(out_path)
    plot_runtime_scatter_vs_constraints(out_path)
    plot_eig_ratio_vs_m_assoc(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start = time.time()
    df = run_analysis()

    print("\n\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    # Persist to CSV for later inspection
    out_path = DEFAULT_CSV
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    generate_plots(out_path)
    print(f"Total elapsed time: {time.time() - start:.2f}s")
