"""Post-processing for the stereo pipeline experiment.

This module houses the post-processing / analysis functions for the results
produced by :mod:`stereo_pipeline_experiment`. That experiment writes results to

    results/stereo_loc/<experiment_name>/<timestamp>/results.csv

alongside the ``experiment.yaml`` and ``stereo_pipeline.yaml`` configs used to
generate them. The loaders below discover those CSVs and read them into
(annotated) pandas DataFrames.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "stereo_loc"


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
        data_dir: Root ``results/stereo_loc`` directory to search.

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


def plot_cert_da_success_rate(
    df: pd.DataFrame, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot the ``cert_da`` success rate vs. ``inv_mult`` for each frame interval.

    For every ``(frame_interval, inv_mult)`` pair, computes the percent of trials
    with ``cert_da`` true and draws one line per ``frame_interval`` as a function
    of ``inv_mult`` (log-scaled x-axis).
    """
    if ax is None:
        _, ax = plt.subplots()

    # cert_da is typically read as bool, but guard against string "True"/"False".
    cert_da = df["cert_da"]
    if cert_da.dtype == object:
        cert_da = cert_da.astype(str).str.strip().str.lower() == "true"

    rates = (
        cert_da.groupby([df["frame_interval"], df["inv_mult"]])
        .mean()
        .mul(100.0)
        .rename("cert_da_pct")
        .reset_index()
    )

    for frame_interval, group in rates.groupby("frame_interval"):
        group = group.sort_values("inv_mult")
        ax.plot(
            group["inv_mult"],
            group["cert_da_pct"],
            marker="o",
            label=f"frame_interval = {frame_interval}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("inv_mult (log scale)")
    ax.set_ylabel("cert_da success rate (%)")
    ax.set_title("Certificate (cert_da) success rate vs. invariant multiplier")
    ax.set_ylim(-2, 102)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Frame interval")
    return ax


def plot_pose_error(
    df: pd.DataFrame, axes: Optional[List[plt.Axes]] = None
) -> List[plt.Axes]:
    """Plot mean translation and rotation error vs. ``inv_mult``.

    Draws two panels (translation error, rotation error) sharing an ``inv_mult``
    (log-scaled) x-axis, with one line per ``frame_interval``.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        ("err_trans", "Mean translation error", axes[0]),
        ("err_rot", "Mean rotation error", axes[1]),
    ]
    for col, title, ax in panels:
        means = (
            df.groupby(["frame_interval", "inv_mult"])[col]
            .mean()
            .rename(col)
            .reset_index()
        )
        for frame_interval, group in means.groupby("frame_interval"):
            group = group.sort_values("inv_mult")
            ax.plot(
                group["inv_mult"],
                group[col],
                marker="o",
                label=f"frame_interval = {frame_interval}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("inv_mult (log scale)")
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="Frame interval")
    return list(axes)


def invariant_sweep_results():
    """Load and plot the results of the invariant sweep experiment."""
    df = load_results(experiment_name="invariant_sweep", timestamp="latest")
    print(f"\nLoaded {len(df)} rows from {df['timestamp'].nunique()} run(s).")
    print(df.head())

    plot_cert_da_success_rate(df)
    plt.tight_layout()

    plot_pose_error(df)
    plt.tight_layout()

    plt.show()


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a certification column to bool, tolerating string "True"/"False"."""
    if series.dtype == object:
        return series.astype(str).str.strip().str.lower() == "true"
    return series.astype(bool)


def machine_hall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise Machine Hall results with one row per ``frame_interval``.

    Adds an ``outlier_ratio`` column (``num_inliers / num_valid``) and aggregates
    per ``frame_interval`` into a table of average time interval, average number
    of inliers, average outlier rate, and the DA / registration certification
    rates (percent of trials with ``cert_da`` / ``cert_reg`` true).
    """
    df = df.copy()
    df["outlier_ratio"] = 1.0 - df["num_inliers"] / df["num_valid"]
    df["err_rot_deg"] = np.degrees(df["err_rot"])
    df["cert_da"] = _as_bool(df["cert_da"])
    df["cert_reg"] = _as_bool(df["cert_reg"])

    summary = (
        df.groupby("frame_interval")
        .agg(
            avg_time_interval=("time_interval", "mean"),
            avg_num_inliers=("num_inliers", "mean"),
            avg_outlier_rate=("outlier_ratio", "mean"),
            avg_err_trans=("err_trans", "mean"),
            avg_err_rot_deg=("err_rot_deg", "mean"),
            da_cert_rate=("cert_da", "mean"),
            reg_cert_rate=("cert_reg", "mean"),
        )
        .reset_index()
    )
    summary["da_cert_rate"] = summary["da_cert_rate"].mul(100.0)
    summary["reg_cert_rate"] = summary["reg_cert_rate"].mul(100.0)
    summary["avg_outlier_rate"] = summary["avg_outlier_rate"].mul(100.0)
    summary.drop(columns="frame_interval", inplace=True)
    return summary


def _split_by_both_cert(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    """Average ``value_cols`` per ``frame_interval``, split by ``both_cert``.

    ``both_cert`` is true when both ``cert_da`` and ``cert_reg`` are true. Each
    value column becomes two columns suffixed ``_cert`` (both certificates
    passed) and ``_uncert`` (at least one failed). Rows are labelled by the
    average time interval of the frame interval rather than the frame interval
    itself.
    """
    df = df.copy()
    df["both_cert"] = _as_bool(df["cert_da"]) & _as_bool(df["cert_reg"])

    means = df.groupby(["frame_interval", "both_cert"])[value_cols].mean()

    stats = means.unstack("both_cert")
    # Flatten the (column, both_cert) MultiIndex into "<col>_cert"/"<col>_uncert".
    stats.columns = [
        f"{col}_{'cert' if flag else 'uncert'}" for col, flag in stats.columns
    ]
    ordered = []
    for col in value_cols:
        ordered += [f"{col}_cert", f"{col}_uncert"]
    ordered = [c for c in ordered if c in stats.columns]
    stats = stats[ordered].reset_index()

    avg_time = (
        df.groupby("frame_interval")["time_interval"].mean().rename("avg_time_interval")
    )
    stats = stats.merge(avg_time, on="frame_interval").drop(columns="frame_interval")
    # Lead with avg_time_interval.
    return stats[["avg_time_interval"] + ordered]


def machine_hall_error_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average translation / rotation error per ``frame_interval``, split on cert.

    Rotation error is reported in degrees (``err_rot_deg``). The split is on
    whether both ``cert_da`` and ``cert_reg`` are true for a trial (``_cert``)
    versus not (``_uncert``).
    """
    df = df.copy()
    df["err_rot_deg"] = np.degrees(df["err_rot"])
    return _split_by_both_cert(df, ["err_trans", "err_rot_deg"])


def machine_hall_runtime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average certification runtimes per ``frame_interval``, in milliseconds.

    Reports ``cert_time_da_ms`` / ``cert_time_reg_ms``. Split on whether both
    ``cert_da`` and ``cert_reg`` are true for a trial (``_cert``) versus not
    (``_uncert``).
    """
    df = df.copy()
    df["cert_time_da_ms"] = df["cert_time_da"] * 1e3
    df["cert_time_reg_ms"] = df["cert_time_reg"] * 1e3
    return _split_by_both_cert(df, ["cert_time_da_ms", "cert_time_reg_ms"])


def machine_hall_results():
    """Load and summarise the Machine Hall (MH01e_clipper_reduced) results."""
    df = load_results(experiment_name="MH01e_clipper_reduced", timestamp="latest")
    print(f"\nLoaded {len(df)} rows from {df['timestamp'].nunique()} run(s).")

    summary = machine_hall_summary(df)
    print("\nMachine Hall summary (per frame interval):")
    print(
        summary.to_string(
            index=False,
            formatters={
                "avg_time_interval": "{:.2f}".format,
                "avg_num_inliers": "{:.1f}".format,
                "avg_outlier_rate": "{:.1f}%".format,
                "avg_err_trans": "{:.4f}".format,
                "avg_err_rot_deg": "{:.3f}".format,
                "da_cert_rate": "{:.1f}%".format,
                "reg_cert_rate": "{:.1f}%".format,
            },
        )
    )

    errors = machine_hall_error_summary(df)
    print("\nPose error split by (cert_da AND cert_reg):")
    print(
        errors.to_string(
            index=False,
            formatters={
                "avg_time_interval": "{:.4f}".format,
                **{
                    c: "{:.6f}".format
                    for c in errors.columns
                    if c.startswith(("err_trans", "err_rot_deg"))
                },
            },
        )
    )

    runtimes = machine_hall_runtime_summary(df)
    print("\nCertification runtimes split by (cert_da AND cert_reg):")
    print(
        runtimes.to_string(
            index=False,
            formatters={
                "avg_time_interval": "{:.4f}".format,
                **{
                    c: "{:.1f}".format
                    for c in runtimes.columns
                    if c.startswith(("cert_time_da_ms", "cert_time_reg_ms"))
                },
            },
        )
    )
    return summary, errors, runtimes


if __name__ == "__main__":
    # invariant_sweep_results()
    machine_hall_results()
