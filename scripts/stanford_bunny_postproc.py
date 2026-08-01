"""Post-processing for the Stanford bunny data association experiment.

This module houses the post-processing / analysis functions for the results
produced by :mod:`standford_bunny_experiment`. That experiment writes results to

    results/data_association/<experiment_name>/<timestamp>/results.csv

alongside the ``experiment.yaml`` config used to generate them. The loaders below
discover those CSVs and read them into (annotated) pandas DataFrames.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.colors as mcolors
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


def _darken(color, factor: float = 0.8):
    """Return ``color`` scaled toward black (same hue, lower brightness)."""
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


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


def load_timing_sweep(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the ``timing_sweep`` experiment results."""
    # Load both the original and SDP-augmented runs and concatenate them.
    df_outlier = load_results("timing_sweep_outlier_ratio", data_dir=data_dir)
    df_assoc = load_results("timing_sweep_num_assoc", data_dir=data_dir)
    return df_outlier, df_assoc


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


def _log_grouped_boxplot(ax, cats, series, cluster_frac=0.8, axes=None):
    """Draw side-by-side box plots at continuous, log-scaled x positions.

    ``cats`` are the actual (numeric) x values and ``series`` is a list of
    ``(label, color, values_per_cat)`` tuples where ``values_per_cat[i]`` holds
    the values for ``cats[i]``. Boxes are positioned and sized multiplicatively
    (in log space) so they appear uniform on a log x-axis, and the cluster for
    each category is centered on its value. The x-axis is set to log scale with
    a tick at every decade.

    ``axes`` optionally gives a per-series target axis (list parallel to
    ``series``); when provided, series ``j`` is drawn on ``axes[j]`` instead of
    ``ax``. Slot positions are still computed over the full ``series`` list, so
    boxes drawn on different (e.g. twinned) axes do not overlap in x.
    """
    cats = np.asarray(sorted(cats), dtype=float)
    logc = np.log10(cats)
    # Smallest gap between categories (in decades) sets the cluster width.
    dlog = float(np.min(np.diff(logc))) if len(cats) > 1 else 1.0
    slot = dlog * cluster_frac / max(len(series), 1)

    for j, (_, color, per_cat) in enumerate(series):
        draw_ax = ax if axes is None else axes[j]
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
        draw_ax.boxplot(
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
            flierprops=dict(
                marker=".",
                markersize=3,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.6,
            ),
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs="auto"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _categorical_grouped_boxplot(ax, cats, series, cluster_frac=0.8):
    """Draw side-by-side box plots at evenly spaced categorical x positions.

    ``cats`` are the (numeric) category values and ``series`` is a list of
    ``(label, color, values_per_cat)`` tuples where ``values_per_cat[i]`` holds
    the values for ``cats[i]``. Categories are placed at integer positions
    ``0..len(cats)-1`` and the cluster of boxes for each category is centered on
    its position; tick labels are set to the category values. Returns a list of
    proxy legend handles (one per series) in ``series`` order.
    """
    cats = list(cats)
    positions_base = np.arange(len(cats))
    slot = cluster_frac / max(len(series), 1)

    handles = []
    for j, (_, color, per_cat) in enumerate(series):
        # Offset of this series within the cluster.
        off = -cluster_frac / 2 + slot * (j + 0.5)
        data, positions = [], []
        for i in range(len(cats)):
            vals = np.asarray(per_cat[i], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            data.append(vals)
            positions.append(positions_base[i] + off)
        handles.append(plt.Line2D([0], [0], color=color, lw=6, alpha=0.6))
        if not data:
            continue
        ax.boxplot(
            data,
            positions=positions,
            widths=slot * 0.9,
            patch_artist=True,
            showfliers=True,
            manage_ticks=False,
            boxprops=dict(facecolor=color, edgecolor=color, alpha=0.6),
            medianprops=dict(color="black"),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(
                marker=".",
                markersize=3,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.6,
            ),
        )

    ax.set_xticks(positions_base)
    ax.set_xticklabels([f"{c:g}" for c in cats])
    ax.set_xlim(-0.5, len(cats) - 0.5)
    return handles


# High-contrast (Okabe-Ito) triad, one colour per timing series.
TIMING_SERIES_COLORS = {
    "Mosek": "#0072B2",  # SDP solve time
    "Clipper": "#E69F00",  # CLIPPER solve time
    "CP-Cert": "#009E73",  # SDP certification time
}


def _timing_series(df, catcol, cats):
    """Build ``(label, color, values_per_cat)`` series for the timing plots.

    Splits ``df`` into the CLIPPER and SDP methods and returns the three series
    plotted by the timing figures: the Mosek (SDP) solve time, the Clipper solve
    time, and the CP-Cert (SDP certification) time, each as a list of value
    arrays aligned with ``cats``.
    """
    df_clipper = df[df["method"] == "CLIPPER"]
    df_sdp = df[df["method"] == "SDP"]

    def by_cat(sub, col):
        return [sub.loc[sub[catcol] == c, col].values for c in cats]

    return [
        ("Mosek", TIMING_SERIES_COLORS["Mosek"], by_cat(df_sdp, "t_solver")),
        ("CP-Cert", TIMING_SERIES_COLORS["CP-Cert"], by_cat(df_sdp, "t_certify")),
        ("Clipper", TIMING_SERIES_COLORS["Clipper"], by_cat(df_clipper, "t_solver")),
    ]


def _add_constraints_axis(ax, cats, df, catcol, positions=None):
    """Add a second x-axis reporting the mean constraint count per category.

    ``positions`` are the x locations (in the parent axis' data coordinates) at
    which to place the constraint labels; defaults to the evenly spaced integer
    positions ``0..len(cats)-1``.
    """
    if positions is None:
        positions = np.arange(len(cats))
    means = [df.loc[df[catcol] == c, "num_constraints"].mean() for c in cats]
    sec = ax.secondary_xaxis(-0.22)
    sec.set_xticks(positions)
    sec.set_xticklabels(
        [f"{m:.0f}" for m in means], rotation=45, ha="right", rotation_mode="anchor"
    )
    sec.set_xlabel("Number of Constraints")
    sec.tick_params(length=0)
    return sec


def _categorical_line_plot(ax, cats, series, xpos=None):
    """Draw a mean line per series with a shaded min-max band.

    ``cats`` and ``series`` follow the same convention as
    :func:`_categorical_grouped_boxplot`. Each series is a
    ``(label, color, values_per_cat)`` tuple; for each series a line traces the
    per-category mean and a translucent band spans the per-category [min, max]
    range. Returns a list of proxy legend handles (one per series).

    By default categories are placed at evenly spaced integer positions
    ``0..len(cats)-1``. If ``xpos`` is given, the series are instead plotted at
    those x locations (e.g. the actual category values) and the x-axis is set to
    log scale, with major ticks labelled by the category values.
    """
    log_x = xpos is not None
    positions = np.arange(len(cats)) if xpos is None else np.asarray(xpos, dtype=float)

    # First pass: reduce each series to per-x-value (mean, min, max) points.
    def _reduce(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan, np.nan, np.nan
        return vals.mean(), vals.min(), vals.max()

    reduced = []  # (label, color, means, mins, maxs) per series
    for label, color, per_cat in series:
        stats = np.array([_reduce(per_cat[i]) for i in range(len(cats))])
        means, mins, maxs = stats[:, 0], stats[:, 1], stats[:, 2]
        reduced.append((label, color, means, mins, maxs))

    # Second pass: plot the mean line and min-max band from the reduced points.
    handles = []
    for label, color, means, mins, maxs in reduced:
        ax.fill_between(positions, mins, maxs, color=color, alpha=0.25, linewidth=0)
        (line,) = ax.plot(
            positions, means, "-o", color=color, markersize=4, label=label
        )
        handles.append(line)

    # Set the log scale before the ticks so it doesn't reset the fixed locator.
    if log_x:
        ax.set_xscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{c:g}" for c in cats])
    if log_x:
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.set_xlim(-0.5, len(cats) - 0.5)
    return handles


def plot_timing_sweep_boxplots(
    df_outlier: pd.DataFrame,
    df_assoc: pd.DataFrame,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """Box plots of solver/certifier runtimes for the timing sweeps.

    Produces a two-panel figure. The left panel sweeps the outlier ratio (fixed
    number of associations) and the right panel sweeps the number of associations
    (fixed outlier ratio). Each panel shows three side-by-side box plots per x
    value on a log y-axis: the SDP solve time, the CLIPPER solve time, and the
    SDP certification time. ``df_outlier`` and ``df_assoc`` are expected to
    contain both the ``CLIPPER`` and ``SDP`` methods.

    Returns the created ``(fig, axes)``.
    """
    fig, (ax_out, ax_assoc) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    out_cats = sorted(df_outlier["outlier_ratio"].unique())
    out_series = _timing_series(df_outlier, "outlier_ratio", out_cats)
    handles = _categorical_grouped_boxplot(ax_out, out_cats, out_series)
    ax_out.set_xlabel(
        f"Outlier Ratio (Num. Assoc. = {df_outlier['num_assoc'].iloc[0]})"
    )
    ax_out.set_ylabel("Runtime (s)")
    ax_out.set_yscale("log")
    ax_out.grid(True, which="both", axis="y", alpha=0.3)
    ax_out.legend(handles, [label for label, _, _ in out_series])
    _add_constraints_axis(ax_out, out_cats, df_outlier, "outlier_ratio")

    assoc_cats = sorted(df_assoc["num_assoc"].unique())
    assoc_series = _timing_series(df_assoc, "num_assoc", assoc_cats)
    _categorical_grouped_boxplot(ax_assoc, assoc_cats, assoc_series)
    ax_assoc.set_xlabel(
        "Number of Associations (Outlier Ratio = "
        f"{df_assoc['outlier_ratio'].iloc[0]:.2f})"
    )
    ax_assoc.set_yscale("log")
    ax_assoc.grid(True, which="both", axis="y", alpha=0.3)
    ax_assoc.tick_params(labelleft=True)
    _add_constraints_axis(ax_assoc, assoc_cats, df_assoc, "num_assoc")

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    return fig, (ax_out, ax_assoc)


def plot_timing_sweep_lines(
    df_outlier: pd.DataFrame,
    df_assoc: pd.DataFrame,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """Line-plot version of :func:`plot_timing_sweep_boxplots`.

    Same two-panel layout, secondary constraint-count axis, and three series
    (Mosek / Clipper / CP-Cert) on a log y-axis, but each box cluster is replaced
    by a mean line with a translucent band spanning the [min, max] runtime range
    across trials. ``df_outlier`` and ``df_assoc`` are expected to contain both
    the ``CLIPPER`` and ``SDP`` methods.

    Returns the created ``(fig, axes)``.
    """
    fig_scale = 1.2
    fig, (ax_assoc, ax_out) = plt.subplots(2, 1, figsize=(5 * fig_scale, 5 * fig_scale))

    out_cats = sorted(df_outlier["outlier_ratio"].unique())
    out_series = _timing_series(df_outlier, "outlier_ratio", out_cats)
    # Plot against the actual outlier ratios on a log x-axis.
    out_arr = np.asarray(out_cats, dtype=float)
    _categorical_line_plot(ax_out, out_cats, out_series, xpos=out_arr)
    ax_out.set_xlabel(
        f"Outlier Ratio (Num. Assoc. = {df_outlier['num_assoc'].iloc[0]})"
    )
    ax_out.set_ylabel("Runtime (s)")
    ax_out.set_yscale("log")
    ax_out.grid(True, which="both", axis="y", alpha=0.3)

    _add_constraints_axis(
        ax_out, out_cats, df_outlier, "outlier_ratio", positions=out_arr
    )

    assoc_cats = sorted(df_assoc["num_assoc"].unique())
    assoc_series = _timing_series(df_assoc, "num_assoc", assoc_cats)
    # Plot against the actual association counts on a log x-axis.
    assoc_arr = np.asarray(assoc_cats, dtype=float)
    handles = _categorical_line_plot(ax_assoc, assoc_cats, assoc_series, xpos=assoc_arr)
    # Cubic reference line (grey dashed, no markers, excluded from legend),
    # anchored at (200, 1): y = (num_assoc / 200) ** 3.
    ax_assoc.plot(
        assoc_arr,
        (assoc_arr / 200.0) ** 3,
        "--",
        color="grey",
        label="_nolegend_",
    )
    ax_assoc.set_xlabel(
        "Number of Associations (Outlier Ratio = "
        f"{df_assoc['outlier_ratio'].iloc[0]:.2f})"
    )
    ax_assoc.set_yscale("log")
    ax_assoc.set_ylabel("Runtime (s)")
    ax_assoc.grid(True, which="both", axis="y", alpha=0.3)
    ax_assoc.tick_params(labelleft=True)
    ax_assoc.legend(handles, [label for label, _, _ in out_series], ncols=3)
    # _add_constraints_axis(
    #     ax_assoc, assoc_cats, df_assoc, "num_assoc", positions=assoc_arr
    # )

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    return fig, (ax_out, ax_assoc)


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
    keys = ["num_assoc", "outlier_ratio", "trial", "inv_mult"]
    # Only keep trials where the reference method produced a valid num_inliers;
    # a NaN there means the reference solve failed and the trial is invalid.
    valid_keys = df.loc[
        (df["method"] == reference_method) & df["num_inliers"].notna(), keys
    ]
    df = df.merge(valid_keys.drop_duplicates(), on=keys, how="inner")

    # Keep only run-keys that have a row for every method present in df.
    n_methods = df["method"].nunique()
    df = df.groupby(keys).filter(lambda g: g["method"].nunique() == n_methods)

    ref = df[df["method"] == reference_method][keys + ["obj_value"]].rename(
        columns={"obj_value": "obj_value_ref"}
    )
    merged = df.merge(ref, on=keys, how="inner")

    merged["gap"] = (merged["obj_value"] - merged["obj_value_ref"]) / (
        merged["obj_value_ref"].abs()
    )

    merged["gt_cert"] = merged["gap"] < threshold
    pred = merged["cert_da"].astype(bool)
    gt = merged["gt_cert"]

    merged["TP"] = pred & gt
    merged["FP"] = pred & ~gt
    merged["TN"] = ~pred & ~gt
    merged["FN"] = ~pred & gt

    counts = merged.groupby("method")[["TP", "FP", "TN", "FN"]].sum()
    n = counts.sum(axis=1)
    table = counts.div(n, axis=0) * 100.0
    table["N"] = n
    return table


def plot_invariant_sweep_boxplots(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """Box-plot version of :func:`plot_invariant_sweep`.

    Produces a three-panel figure (shared ``inv_mult`` x-axis). The invariant
    multiplier is treated as a continuous variable: boxes sit at their actual
    value on a log-scaled x-axis with a tick at every decade. The box plots show
    only the CLIPPER method; the certified-trials panel covers every method in
    ``df``. The top panel shows side-by-side box plots of the relative
    translation and rotation error within each invariant multiplier, split by
    whether the trial's data association was certified (``cert_da``). Translation
    error (m) is on the left log y axis and rotation error (deg) on a twinned
    right log y axis. The middle panel shows, per method, the percent of trials certified
    (solid) and the percent cost-certified against the SDP optimum (dashed,
    darker), overlaid with the SDP rank-tight rate. The bottom panel shows box
    plots of the percent inliers found per invariant multiplier.

    Returns the created ``(fig, axes)``.
    """
    df_clipper = df[df["method"] == "CLIPPER"]
    df_sdp = df[df["method"] == "SDP"]
    cats = sorted(df_clipper["inv_mult"].unique())

    # 2x2 layout sized to fit the top third of a standard paper page: it spans
    # the full text width (~7 in) with a height of ~1/3 of the text area.
    fig_scale = 2.2
    fig, axes = plt.subplots(
        2, 2, figsize=(6 * fig_scale, 2.5 * fig_scale), sharex=True
    )
    (ax_corres, ax_cert), (ax_reg_err, ax_none) = axes

    # --- Registration error distributions (log scale) ---
    # Translation and rotation error are shown in separate panels, with boxes
    # split by whether the trial's data association was certified.
    def _err_by_cert(col, certified):
        mask = df_clipper["cert_da"] if certified else ~df_clipper["cert_da"]
        return [
            df_clipper.loc[(df_clipper["inv_mult"] == c) & mask, col].values
            for c in cats
        ]

    trans_series = [
        ("Trans. (Cert.)", "#12428A", _err_by_cert("trans_error", True)),
        ("Trans. (No Cert.)", "#20C0D0", _err_by_cert("trans_error", False)),
    ]
    rot_series = [
        ("Rot. (Cert.)", "#C0202B", _err_by_cert("rot_error_deg", True)),
        ("Rot. (No Cert.)", "#F58BB0", _err_by_cert("rot_error_deg", False)),
    ]
    # Translation and rotation carry different units, so they get separate y
    # axes on the same panel: translation (m) on the left, rotation (deg) on a
    # twinned right axis. Slots are computed over the combined series list so the
    # two axes' boxes interleave without overlapping in x.
    ax_rot = ax_reg_err.twinx()
    err_series = trans_series + rot_series
    err_axes = [ax_reg_err, ax_reg_err, ax_rot, ax_rot]
    _log_grouped_boxplot(ax_reg_err, cats, err_series, axes=err_axes)
    ax_reg_err.set_yscale("log")
    ax_rot.set_yscale("log")
    ax_reg_err.set_ylabel("Translation Error (m)")
    ax_rot.set_ylabel("Rotation Error (deg)")
    ax_reg_err.grid(True, which="both", alpha=0.3)
    handles = [
        plt.Line2D([0], [0], color=color, lw=6, alpha=0.6) for _, color, _ in err_series
    ]
    ax_reg_err.legend(handles, [label for label, _, _ in err_series])
    ax_none.set_visible(False)

    # --- Percent of trials certified (boolean), per method ---
    # For each method: a solid line shows the percent of trials whose data
    # association was certified (``cert_da``), and a dashed, slightly darker
    # line (same hue) shows the percent that were cost-certified -- i.e. whose
    # objective matches the SDP optimum to within COST_GAP_TOL. Only the solid
    # per-method lines and the SDP rank-tight line get legend entries (five
    # total); the dashed cost-certified lines share their method's color.
    COST_GAP_TOL = 1e-4
    keys = ["outlier_ratio", "num_assoc", "trial", "inv_mult"]
    ref = df[df["method"] == "SDP"][keys + ["obj_value"]].rename(
        columns={"obj_value": "obj_value_sdp"}
    )
    merged = df.merge(ref, on=keys, how="inner")
    merged["cost_cert"] = (
        (merged["obj_value"] - merged["obj_value_sdp"]) / merged["obj_value_sdp"]
    ).abs() < COST_GAP_TOL

    for method in sorted(df["method"].unique()):
        if method == "SDP":
            continue

        color = METHOD_COLORS.get(method)
        cert_pct = (
            df[df["method"] == method]
            .groupby("inv_mult")["cert_da"]
            .mean()
            .mul(100.0)
            .reindex(cats)
        )
        ax_cert.plot(cats, cert_pct.values, "-", color=color, label=method)
        cost_cert = (
            merged[merged["method"] == method]
            .groupby("inv_mult")["cost_cert"]
            .mean()
            .mul(100.0)
            .reindex(cats)
        )
        ax_cert.plot(cats, cost_cert.values, "--", color=_darken(color, 0.7))

    # SDP rank-tight rate: fraction of SDP trials with a rank-tight solution.
    rank_tight = (
        df_sdp.groupby("inv_mult")["num_inliers"]
        .apply(lambda s: s.notna().mean() * 100.0)
        .reindex(cats)
    )
    ax_cert.plot(cats, rank_tight.values, "--", color="#8172B3", label="SDP Rank-Tight")

    ax_cert.set_ylabel("Percent of Trials (%)")
    # ax_cert.set_ylim(0, 101)
    ax_cert.grid(True, which="both", alpha=0.3)
    ax_cert.legend()

    # --- Percent inliers distribution + certified-trial line ---
    # Two boxes per inv_mult: the CLIPPER and SDP accepted-correspondence rates.
    inlier_series = [
        (
            "Accptd. Corresp. (Clipper)",
            "#55A868",
            [
                df_clipper.loc[df_clipper["inv_mult"] == c, "percent_inliers"].values
                for c in cats
            ],
        ),
        (
            "Accptd. Corresp. (SDP)",
            "#2E6B45",
            (
                [
                    df_sdp.loc[df_sdp["inv_mult"] == c, "percent_inliers"].values
                    for c in cats
                ]
                if df_sdp is not None
                else [np.array([]) for _ in cats]
            ),
        ),
    ]
    recall_series = [
        (
            "Recall",
            "#C9261B",
            [
                df_clipper.loc[df_clipper["inv_mult"] == c, "recall"].values * 100
                for c in cats
            ],
        )
    ]
    precision_series = [
        (
            "Precision",
            "#2522C9",
            [
                df_clipper.loc[df_clipper["inv_mult"] == c, "precision"].values * 100
                for c in cats
            ],
        )
    ]
    _log_grouped_boxplot(ax_corres, cats, inlier_series)
    _log_grouped_boxplot(ax_corres, cats, recall_series)
    _log_grouped_boxplot(ax_corres, cats, precision_series)
    ax_corres.set_ylabel("Percent (%)")
    ax_corres.set_ylim(0, 101)
    ax_corres.grid(True, which="both", alpha=0.3)
    # reference line for number of ground-truth correspondences
    ax_corres.axhline(
        y=50, color="grey", linestyle="--", label="Ground Truth Correspondences"
    )
    ax_corres.text(
        x=0.007, y=51, s="Percent True \nCorrespondences", color="grey", va="bottom"
    )

    # Reconcile the box-plot legend with the certified-trials line.
    handles = [
        plt.Line2D([0], [0], color="#55A868", lw=6, alpha=0.6),
        plt.Line2D([0], [0], color="#2E6B45", lw=6, alpha=0.6),
        plt.Line2D([0], [0], color="#C9261B", lw=6, alpha=0.6),
        plt.Line2D([0], [0], color="#2522C9", lw=6, alpha=0.6),
    ]
    ax_corres.legend(
        handles,
        [
            "Corresp. (Clipper)",
            "Corresp. (Global)",
            "Recall",
            "Precision",
        ],
    )
    x_ax_string = r"Noise Ratio Assumed/Actual ($\alpha = \sigma / \gamma$)"
    for ax in (ax_corres, ax_cert, ax_reg_err):
        ax.set_xlabel(x_ax_string)
        ax.tick_params(labelbottom=True)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    return fig, axes


def plot_invariant_sweep_cost_certified(
    df: pd.DataFrame,
    reference_method: str = "SDP",
    cost_gap_tol: float = 1e-4,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """Cost-certified rate per invariant multiplier, for every method.

    Generalizes the "Cost Certified" line of
    :func:`plot_invariant_sweep_boxplots` to all methods present in ``df``. For
    each trial the relative cost gap to the reference (SDP) optimum is

        gap = |(obj_value(method) - obj_value(ref)) / obj_value(ref)|,

    and the trial is cost-certified when ``gap < cost_gap_tol`` (its objective
    effectively matches the SDP optimum). The percent of cost-certified trials is
    plotted against ``inv_mult`` on a log x-axis, one line per method colored by
    :data:`METHOD_COLORS`.

    Returns the created ``(fig, ax)``.
    """
    cats = sorted(df["inv_mult"].unique())
    keys = ["outlier_ratio", "num_assoc", "trial", "inv_mult"]

    # Attach the reference (SDP) objective to every row via its run key, then
    # compute each trial's relative cost gap to that reference optimum.
    ref = df[df["method"] == reference_method][keys + ["obj_value"]].rename(
        columns={"obj_value": "obj_value_ref"}
    )
    merged = df.merge(ref, on=keys, how="inner")
    merged["cost_gap"] = (
        (merged["obj_value"] - merged["obj_value_ref"]) / merged["obj_value_ref"]
    ).abs()
    merged["cost_cert"] = merged["cost_gap"] < cost_gap_tol

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, sub in merged.groupby("method"):
        color = METHOD_COLORS.get(method)
        cost_cert = sub.groupby("inv_mult")["cost_cert"].mean().mul(100.0).reindex(cats)
        ax.plot(
            cats,
            cost_cert.values,
            "-o",
            markersize=4,
            color=color,
            label=method,
        )
        # Dashed line: percent of trials where the data-association was certified
        # (``cert_da`` true), same color as the method's solid cost-certified line.
        cert_da = sub.groupby("inv_mult")["cert_da"].mean().mul(100.0).reindex(cats)
        ax.plot(
            cats,
            cert_da.values,
            "--",
            color=color,
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs="auto"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(r"Noise Ratio Assumed/Actual ($\alpha = \sigma / \gamma$)")
    ax.set_ylabel("Percent of Trials Cost Certified (%)")
    ax.set_ylim(0, 101)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    return fig, ax


def invariant_sweep_results(timestamp=None):
    df = load_results("invariant_sweep", timestamp=timestamp)
    print(f"\nLoaded {len(df)} rows from {df['timestamp'].nunique()} run(s).")

    # Get percent inliers
    df["percent_inliers"] = df["num_inliers"] / df["num_assoc"] * 100.0

    # Registration error from the se(3) pose-error vector stored per trial
    # (xi_err_0..5, translation part first then rotation part, pylgmath order).
    # Translation error is the norm of the translation block (metres); rotation
    # error is the norm of the rotation block converted to degrees.
    trans_cols = ["xi_err_0", "xi_err_1", "xi_err_2"]
    rot_cols = ["xi_err_3", "xi_err_4", "xi_err_5"]
    df["trans_error"] = np.linalg.norm(df[trans_cols].values, axis=1)
    df["rot_error_deg"] = np.degrees(np.linalg.norm(df[rot_cols].values, axis=1))

    plot_invariant_sweep_boxplots(
        df,
        save_path=DATA_DIR
        / "invariant_sweep"
        / "figures"
        / "invariant_sweep_boxplots.png",
    )

    # Cost-certified rate across every method (generalizes the "Cost Certified"
    # line of the box-plot figure to all methods in the sweep).
    plot_invariant_sweep_cost_certified(
        df,
        save_path=DATA_DIR
        / "invariant_sweep"
        / "figures"
        / "invariant_sweep_cost_certified.png",
    )

    threshold = 1e-4
    print(f"\nCertifier confusion vs. SDP ground truth (gap threshold {threshold:g}):")
    confusion = certifier_confusion(df, threshold=threshold)
    print(confusion.to_string(float_format="%.2f"))
    print(confusion.to_latex(float_format="%.2f"))


def timing_sweep_results():
    df_outlier, df_assoc = load_timing_sweep()
    print(
        f"\nLoaded {len(df_outlier)} rows from {df_outlier['timestamp'].nunique()} run(s)."
    )
    print(f"Loaded {len(df_assoc)} rows from {df_assoc['timestamp'].nunique()} run(s).")

    threshold = 1e-4
    print(f"\nOulier sweep confusion (gap threshold {threshold:g}):")
    confusion = certifier_confusion(df_outlier, threshold=threshold)
    print(confusion.to_string(float_format="%.2f"))
    print(f"\nAssoc sweep confusion (gap threshold {threshold:g}):")
    confusion = certifier_confusion(df_assoc, threshold=threshold)
    print(confusion.to_string(float_format="%.2f"))

    # Keep only the CLIPPER and SDP methods; the other methods are not plotted.
    methods = ["CLIPPER", "SDP"]
    df_outlier = df_outlier[df_outlier["method"].isin(methods)]
    df_assoc = df_assoc[df_assoc["method"].isin(methods)]

    df = pd.concat([df_outlier, df_assoc], ignore_index=True)
    print(
        f"Average number of iterations for certification:{df[df["method"]=="CLIPPER"]["num_iter_cert"].mean()}"
    )

    plot_timing_sweep_boxplots(
        df_outlier,
        df_assoc,
        save_path=DATA_DIR / "timing_sweep" / "figures" / "timing_sweep_boxplots.png",
    )

    plot_timing_sweep_lines(
        df_outlier,
        df_assoc,
        save_path=DATA_DIR / "timing_sweep" / "figures" / "timing_sweep_lines.png",
    )


if __name__ == "__main__":

    # Genreate the timing sweep experiment results
    # timing_sweep_results()

    # Generate invariant parameter sweep experiment results
    invariant_sweep_results(timestamp="20260801T0031")
