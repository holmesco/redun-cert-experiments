"""Post-processing for the pose registration certification experiment.

This module houses the post-processing / analysis functions for the results
produced by :mod:`pose_reg_experiment`. That experiment writes results to

    results/pose_registration/<experiment_name>/<timestamp>/results.csv

alongside the ``experiment.yaml`` config used to generate them. The loaders below
discover those CSVs and read them into (annotated) pandas DataFrames.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "pose_registration"


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
        data_dir: Root ``results/pose_registration`` directory to search.

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


def certifier_summary_by_distance(
    df: pd.DataFrame,
    distance_col: str = "camera_distance",
) -> pd.DataFrame:
    """Confusion table and certification timings, aggregated per camera distance.

    The certifier's prediction is ``cert_reg`` and the ground truth is
    ``global_min`` (whether the trial's cost matched the reference SDP cost to
    within the experiment's relative tolerance). For each value of
    ``distance_col`` these are tabulated as

        TP: cert_reg and global_min           FP: cert_reg and not global_min
        FN: not cert_reg and global_min       TN: not cert_reg and not global_min

    reported as a percent of the trials at that distance. The table also reports
    the mean certification time (``t_certify``) split by whether the trial was
    certified, and the mean interior point solve time (``t_sdp``).

    Returns a DataFrame indexed by ``distance_col`` with columns
    ``[TP, FP, TN, FN, N, t_sdp, t_cert_true, t_cert_false]`` where the
    certification timing columns are NaN when no trial falls in that group.
    ``t_sdp`` is omitted for runs predating that column.
    """
    df = df.copy()
    pred = df["cert_reg"].astype(bool)
    gt = df["global_min"].astype(bool)

    df["TP"] = pred & gt
    df["FP"] = pred & ~gt
    df["TN"] = ~pred & ~gt
    df["FN"] = ~pred & gt

    counts = df.groupby(distance_col)[["TP", "FP", "TN", "FN"]].sum()
    n = counts.sum(axis=1)
    table = counts.div(n, axis=0) * 100.0
    table["N"] = n

    # The SDP is solved once per instance and its time repeated across that
    # instance's trials, so this is a trial-weighted mean over the instances.
    if "t_sdp" in df:
        table["t_sdp"] = df.groupby(distance_col)["t_sdp"].mean() * 1000

    # Mean certification time, split by the certifier's own verdict.
    t_cert = df.groupby([distance_col, pred.rename("cert_reg")])["t_certify"].mean()
    t_cert = t_cert.unstack("cert_reg") * 1000  # convert to ms
    table["t_cert_true"] = t_cert.get(True)
    table["t_cert_false"] = t_cert.get(False)
    return table


def pose_reg_results(experiment_name: Optional[str] = None, timestamp=None):
    df = load_results(experiment_name, timestamp=timestamp)
    print(f"\nLoaded {len(df)} rows from {df['timestamp'].nunique()} run(s).")

    print("\nCertifier summary by camera distance:")
    table = certifier_summary_by_distance(df)
    table.drop(columns=["N"], inplace=True)  # N is not needed in the printed table
    print(table.to_string(float_format="%.1f"))
    print(table.to_latex(float_format="%.1f"))
    return table


if __name__ == "__main__":
    pose_reg_results(experiment_name="distance_sweep", timestamp="latest")
