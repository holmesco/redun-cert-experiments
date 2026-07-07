"""Post-processing for the Stanford bunny data association experiment.

This module houses the post-processing / analysis functions for the results
produced by :mod:`standford_bunny_experiment`. That experiment writes results to

    results/data_association/<experiment_name>/<timestamp>/results.csv

alongside the ``experiment.yaml`` config used to generate them. The loaders below
discover those CSVs and read them into (annotated) pandas DataFrames.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "data_association"

# Fixed color per method so plots are consistent across figures.
METHOD_COLORS = {
    "CLIPPER": "#4C72B0",
    "PMC": "#DD8452",
    "SDP": "#55A868",
    "RANSAC": "#C44E52",
}


def _load_run_csv(csv_path: Path) -> pd.DataFrame:
    """Load a single ``results.csv`` and annotate it with its run metadata.

    Adds ``experiment_name`` and ``timestamp`` columns inferred from the path
    (``.../<experiment_name>/<timestamp>/results.csv``) so that DataFrames from
    multiple runs can be concatenated without losing provenance.
    """
    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    df["timestamp"] = csv_path.parent.name
    df["experiment_name"] = csv_path.parent.parent.name
    return df


def load_results(
    experiment_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Load experiment result CSVs into a single DataFrame.

    Args:
        experiment_name: If given, only load runs under this experiment folder.
            Otherwise load runs from every experiment in ``data_dir``.
        timestamp: If given, load only this run's timestamp. If ``"latest"``,
            load only the most recent run per experiment. Otherwise load all
            timestamped runs.
        data_dir: Root ``results/data_association`` directory to search.

    Returns:
        A concatenated DataFrame with ``experiment_name`` and ``timestamp``
        columns identifying the run each row came from.

    Raises:
        FileNotFoundError: If no matching ``results.csv`` files are found.
    """
    csv_paths = find_result_csvs(experiment_name, timestamp, data_dir)
    if not csv_paths:
        raise FileNotFoundError(
            f"No results.csv found under {data_dir} "
            f"(experiment_name={experiment_name!r}, timestamp={timestamp!r})."
        )
    frames = [_load_run_csv(p) for p in csv_paths]
    return pd.concat(frames, ignore_index=True)


def load_benchmark_sweep_low(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the ``benchmark_sweep_low`` experiment results.

    """
    # Load both the original and SDP-augmented runs and concatenate them.
    df = load_results("benchmark_sweep_low", data_dir=data_dir)
    df_sdp = load_results("benchmark_sweep_low_sdp", data_dir=data_dir)
    df = pd.concat([df, df_sdp], ignore_index=True)
    return df
    


def cert_da_percent_by_method(df: pd.DataFrame) -> pd.Series:
    """Percent of trials with ``cert_da == True`` for each method.

    Returns a Series indexed by ``method`` with values in [0, 100].
    """
    return df.groupby("method")["cert_da"].mean().mul(100.0)


def find_result_csvs(
    experiment_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    data_dir: Path = DATA_DIR,
) -> List[Path]:
    """Discover ``results.csv`` paths matching the given filters.

    See :func:`load_results` for the meaning of the arguments. Returned paths are
    sorted by (experiment_name, timestamp).
    """
    if not data_dir.exists():
        return []

    exp_dirs = (
        [data_dir / experiment_name]
        if experiment_name is not None
        else sorted(d for d in data_dir.iterdir() if d.is_dir())
    )

    csv_paths: List[Path] = []
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        run_dirs = sorted(d for d in exp_dir.iterdir() if d.is_dir())
        if timestamp == "latest":
            run_dirs = run_dirs[-1:]
        elif timestamp is not None:
            run_dirs = [d for d in run_dirs if d.name == timestamp]
        for run_dir in run_dirs:
            csv_path = run_dir / "results.csv"
            if csv_path.is_file():
                csv_paths.append(csv_path)
    return csv_paths

def _grouped_boxplot(ax, x_col, x_categories, series, x_labels=None, box_width=0.8):
    """Draw side-by-side box plots of several series within each x category.

    ``series`` is a list of ``(label, color, value_col, sub_df)`` tuples; each
    contributes one box per category, offset within the cluster so the series can
    be compared directly. ``x_col`` must hold discrete/binned categories present
    in each ``sub_df``.
    """
    n_series = len(series)
    # Width allotted to one cluster, split evenly among the boxes in it.
    slot = box_width / max(n_series, 1)

    for j, (_, color, value_col, sub) in enumerate(series):
        data, positions = [], []
        for i, cat in enumerate(x_categories):
            vals = sub.loc[sub[x_col] == cat, value_col].dropna().values
            if len(vals) == 0:
                continue
            data.append(vals)
            # Offset each series' box within the cluster centered on i.
            positions.append(i - box_width / 2 + slot * (j + 0.5))
        if not data:
            continue
        ax.boxplot(
            data,
            positions=positions,
            widths=slot * 0.9,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, edgecolor=color, alpha=0.6),
            medianprops=dict(color="black"),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )

    ax.set_xticks(range(len(x_categories)))
    ax.set_xticklabels(x_labels if x_labels is not None else x_categories)
    ax.set_ylabel("time [s]")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    # Legend keyed on box color.
    handles = [
        plt.Line2D([0], [0], color=color, lw=6, alpha=0.6)
        for _, color, _, _ in series
    ]
    ax.legend(handles, [label for label, _, _, _ in series])


def _log_grouped_boxplot(ax, cats, series, cluster_frac=0.8):
    """Draw side-by-side box plots at continuous, log-scaled x positions.

    ``cats`` are the actual (numeric) x values and ``series`` is a list of
    ``(label, color, values_per_cat)`` tuples where ``values_per_cat[i]`` holds
    the values for ``cats[i]``. Boxes are positioned and sized multiplicatively
    (in log space) so they appear uniform on a log x-axis, and the cluster for
    each category is centered on its value. The x-axis is set to log scale with
    a tick at every decade.
    """
    cats = np.asarray(sorted(cats), dtype=float)
    logc = np.log10(cats)
    # Smallest gap between categories (in decades) sets the cluster width.
    dlog = float(np.min(np.diff(logc))) if len(cats) > 1 else 1.0
    slot = dlog * cluster_frac / max(len(series), 1)

    for j, (_, color, per_cat) in enumerate(series):
        # Multiplicative offset of this series within the cluster.
        off = -dlog * cluster_frac / 2 + slot * (j + 0.5)
        data, positions, widths = [], [], []
        for i in range(len(cats)):
            vals = np.asarray(per_cat[i])
            vals = vals[~np.isnan(vals)] if vals.dtype.kind == "f" else vals
            if len(vals) == 0:
                continue
            center = 10.0 ** (logc[i] + off)
            data.append(vals)
            positions.append(center)
            # Width in data units that renders as a fixed span in log space.
            widths.append(center * (10.0 ** (slot * 0.45) - 10.0 ** (-slot * 0.45)))
        if not data:
            continue
        ax.boxplot(
            data,
            positions=positions,
            widths=widths,
            patch_artist=True,
            showfliers=True,
            manage_ticks=False,
            boxprops=dict(facecolor=color, edgecolor=color, alpha=0.6),
            medianprops=dict(color="black"),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(marker=".", markersize=3, markerfacecolor=color, markeredgecolor=color, alpha=0.6),
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs="auto"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _solve_certify_series(df):
    """Build the ``_grouped_boxplot`` series for solve + certify.

    Shows ``t_solver`` boxes for SDP and CLIPPER only, plus a single
    ``t_certify`` box for CLIPPER, so these sit in the same cluster.
    """
    solve_methods = [m for m in ("SDP", "CLIPPER") if m in df["method"].unique()]
    series = [
        (f"{m} solve", METHOD_COLORS[m], "t_solver", df[df["method"] == m])
        for m in solve_methods
    ]
    if "CLIPPER" in df["method"].unique():
        series.append(
            ("CLIPPER certify", "#7F7F7F", "t_certify", df[df["method"] == "CLIPPER"])
        )
    return series


def _constraint_bins(df: pd.DataFrame, n_constraint_bins: int):
    """Bucket ``num_constraints`` into log-spaced bins.

    Returns ``(binned_df, cat_idx, bin_labels)`` where ``binned_df`` has an added
    ``_cbin`` integer column, ``cat_idx`` are the occupied bin indices and
    ``bin_labels`` are "lo–hi" range strings for each.
    """
    c = df["num_constraints"]
    edges = np.unique(
        np.geomspace(c.min(), c.max(), n_constraint_bins + 1).round().astype(int)
    )
    # pd.cut assigns each row to a bin; labels are the integer bin index.
    binned = df.assign(_cbin=pd.cut(c, bins=edges, include_lowest=True, labels=False))
    cat_idx = sorted(binned["_cbin"].dropna().unique())
    bin_labels = [f"{edges[i]}–{edges[i + 1]}" for i in cat_idx]
    return binned, cat_idx, bin_labels


def plot_t_certify_boxplots(
    df: pd.DataFrame,
    n_constraint_bins: int = 6,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """Box plots comparing SDP/CLIPPER solve time and CLIPPER certify time.

    Produces a two-panel figure stacked vertically. Both panels overlay, in the
    same cluster, the ``t_solver`` boxes for SDP and CLIPPER plus a CLIPPER
    ``t_certify`` box so solve and certify times can be compared directly. The top panel is grouped
    by ``num_assoc`` (already discrete) and the bottom panel by
    ``num_constraints``, which is nearly continuous so it is bucketed into
    ``n_constraint_bins`` log-spaced bins.

    Returns the created ``(fig, axes)``.
    """
    
    fig, (ax_assoc, ax_constr) = plt.subplots(2, 1, figsize=(12, 11))

    assoc_cats = sorted(df["num_assoc"].unique())
    binned, cat_idx, bin_labels = _constraint_bins(df, n_constraint_bins)

    # --- Top panel: vs num_assoc (discrete) ---
    _grouped_boxplot(ax_assoc, "num_assoc", assoc_cats, _solve_certify_series(df))
    ax_assoc.set_xlabel("num_assoc")
    ax_assoc.set_title("Solve & certify time vs. number of associations")

    # --- Bottom panel: vs num_constraints (binned) ---
    _grouped_boxplot(
        ax_constr, "_cbin", cat_idx, _solve_certify_series(binned), x_labels=bin_labels
    )
    ax_constr.set_xlabel("num_constraints")
    ax_constr.set_title("Solve & certify time vs. number of constraints")
    ax_constr.tick_params(axis="x", labelrotation=30)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    return fig, (ax_assoc, ax_constr)


def certifier_confusion(
    df: pd.DataFrame,
    threshold: float = 1e-4,
    reference_method: str = "SDP",
) -> pd.DataFrame:
    """Confusion table of the certifier against an SDP-derived ground truth.

    For each run keyed by ``(num_assoc, outlier_ratio, trial)`` the relative
    objective gap to the reference (SDP) solution is

        gap = |(obj_value(method) - obj_value(SDP)) / obj_value(SDP)|.

    The ground-truth certificate ``gt_cert`` is ``gap < threshold`` (the solution
    is effectively optimal). This is compared against the certifier's own
    ``cert_da`` prediction to tabulate, per method:

        TP: cert_da and gt_cert           FP: cert_da and not gt_cert
        FN: not cert_da and gt_cert       TN: not cert_da and not gt_cert

    Only runs whose key is present for *every* method are considered, so all
    methods are evaluated on the exact same set of problem instances. Returns a
    DataFrame indexed by method with columns ``[TP, FP, TN, FN, N]``.
    """
    keys = ["num_assoc", "outlier_ratio", "trial"]
    # Keep only run-keys that have a row for every method present in df.
    n_methods = df["method"].nunique()
    df = df.groupby(keys).filter(lambda g: g["method"].nunique() == n_methods)

    ref = (
        df[df["method"] == reference_method][keys + ["obj_value"]]
        .rename(columns={"obj_value": "obj_value_ref"})
    )
    merged = df.merge(ref, on=keys, how="inner")

    merged["gap"] = (merged["obj_value"] - merged["obj_value_ref"]) / (merged["obj_value_ref"]+1).abs()
    
    merged["gt_cert"] = merged["gap"] < threshold
    pred = merged["cert_da"].astype(bool)
    gt = merged["gt_cert"]

    merged["TP"] = pred & gt
    merged["FP"] = pred & ~gt
    merged["TN"] = ~pred & ~gt
    merged["FN"] = ~pred & gt

    table = merged.groupby("method")[["TP", "FP", "TN", "FN"]].sum()
    table["N"] = table.sum(axis=1)
    return table


def plot_invariant_sweep_boxplots(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """Box-plot version of :func:`plot_invariant_sweep`.

    Produces a two-panel figure (shared ``inv_mult`` x-axis). The invariant
    multiplier is treated as a continuous variable: boxes sit at their actual
    value on a log-scaled x-axis with a tick at every decade. The top panel shows
    side-by-side box plots of the relative translation and rotation error within
    each invariant multiplier (log y-axis). The bottom panel shows a box plot of
    the percent inliers found per invariant multiplier, overlaid with a line for
    the percent of trials whose data association was certified (``cert_da``),
    which is boolean and so has no meaningful distribution per group.

    Returns the created ``(fig, axes)``.
    """
    cats = sorted(df["inv_mult"].unique())

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax_err, ax_cert, ax_corres = axes

    # --- Top panel: registration error distributions (log scale) ---
    err_series = [
        (
            "Trans.",
            "#4C72B0",
            [df.loc[df["inv_mult"] == c, "rel_trans_error"].values for c in cats],
        ),
        (
            "Rot.",
            "#DD8452",
            [df.loc[df["inv_mult"] == c, "rel_rot_error"].values for c in cats],
        ),
    ]
    _log_grouped_boxplot(ax_err, cats, err_series)
    ax_err.set_yscale("log")
    ax_err.set_ylabel("Relative Registration Error (%)")
    ax_err.grid(True, which="both", alpha=0.3)
    err_handles = [
        plt.Line2D([0], [0], color=color, lw=6, alpha=0.6)
        for _, color, _ in err_series
    ]
    ax_err.legend(err_handles, [label for label, _, _ in err_series])
    
    # --- Middle panel: percent of trials certified (boolean) ---
    cert_pct = df.groupby("inv_mult")["cert_da"].mean().mul(100.0).reindex(cats)
    ax_cert.plot(cats, cert_pct.values, "-", color="#0C0304")
    ax_cert.set_ylabel("Percent Trials Certified (%)")
    # ax_cert.set_ylim(0, 101)
    ax_cert.grid(True, which="both", alpha=0.3)

    # --- Bottom panel: percent inliers distribution + certified-trial line ---
    inlier_series = [
        (
            "Accptd. Corresp.",
            "#55A868",
            [df.loc[df["inv_mult"] == c, "percent_inliers"].values for c in cats],
        )
    ]
    recall_series = [
        (
            "Recall",
            "#C9261B",
            [df.loc[df["inv_mult"] == c, "recall"].values*100 for c in cats],
        )
    ]
    precision_series = [
        (
            "Precision",
            "#2522C9",
            [df.loc[df["inv_mult"] == c, "precision"].values*100 for c in cats],
        )
    ]
    _log_grouped_boxplot(ax_corres, cats, inlier_series)
    _log_grouped_boxplot(ax_corres, cats, recall_series)
    _log_grouped_boxplot(ax_corres, cats, precision_series)
    ax_corres.set_ylabel("Percent (%)")
    ax_corres.set_ylim(0, 101)
    ax_corres.grid(True, which="both", alpha=0.3)
    # reference line for number of ground-truth correspondences
    ax_corres.axhline(y=50, color='grey', linestyle='--', label='Ground Truth Correspondences')
    ax_corres.text(x=0.007, y=51, s="Percent True \nCorrespondences", color='grey', va='bottom')


    # Reconcile the box-plot legend with the certified-trials line.
    handles = [
        plt.Line2D([0], [0], color="#55A868", lw=6, alpha=0.6),
        plt.Line2D([0], [0], color="#C9261B", lw=6, alpha=0.6),
        plt.Line2D([0], [0], color="#2522C9", lw=6, alpha=0.6),
    ]
    ax_corres.legend(handles, ["Accptd. Corresp.", "Recall", "Precision"])
    ax_corres.set_xlabel("Graph Noise Parameter / Actual Noise Level")

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    return fig, axes


def invariant_sweep_results():
    df = load_results("invariant_sweep", timestamp = "20260707T2030")
    print(f"\nLoaded {len(df)} rows from {df['timestamp'].nunique()} run(s).")
    df['percent_inliers'] = df['num_inliers'] / df['num_assoc'] * 100.0

    summary = df.groupby("inv_mult").agg(
        rel_trans_error=("rel_trans_error", "mean"),
        rel_rot_error=("rel_rot_error", "mean"),
        cert_da_pct=("cert_da", lambda s: s.mean() * 100.0),
        num_inliers=("percent_inliers", "mean"),
    )
    print("\nMetrics by invariant multiplier (mean across trials):")
    print(summary.to_string())

    plot_invariant_sweep_boxplots(
        df,
        save_path=DATA_DIR / "invariant_sweep" / "figures" / "invariant_sweep_boxplots.png",
    )


def benchmark_sweep_low_results():
    df = load_benchmark_sweep_low()
    print(f"\nLoaded {len(df)} rows from {df['timestamp'].nunique()} run(s).")

    print("\nPercent of trials with cert_da == True, by method:")
    cert_pct = cert_da_percent_by_method(df)
    for method, pct in cert_pct.items():
        print(f"  {method:>8}: {pct:5.1f}%")

    threshold = 1e-2
    print(
        f"\nCertifier confusion vs. SDP ground truth (gap threshold {threshold:g}):"
    )
    confusion = certifier_confusion(df, threshold=threshold)
    print(confusion.to_string())
    

    plot_t_certify_boxplots(
        df,
        save_path=DATA_DIR / "benchmark_sweep_low" / "figures" / "t_certify_boxplots.png",
    )



if __name__ == "__main__":
    # benchmark_sweep_low_results()
    invariant_sweep_results()
