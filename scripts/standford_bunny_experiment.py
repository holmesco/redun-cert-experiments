"""Stanford bunny data association experiment.

This experiment stresses the :class:`DataAssociationBlock` family (currently the
CLIPPER block) on synthetic correspondence problems generated from the Stanford
bunny point cloud.

Two experiment types are supported (see :class:`ExperimentType`):

* ``CLIPPER_BENCHMARK`` -- a Python port of the CLIPPER C++ benchmark
  (``extern/clipper/benchmarks/main.cpp``). For a sweep over outlier ratios and
  numbers of associations, we build synthetic putative correspondences from a
  noisy copy of the bunny and measure affinity-matrix construction time, solve
  time, precision and recall of the resulting data association.
* ``ADVERSARIAL`` -- scaffolding only for now; to be described later.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Tuple
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.spatial import KDTree
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation


from stereo_loc.DataAssociationBlocks import (
    DataAssociationBlock,
    DataAssociationConfig,
    DataAssociationMethod,
)

ROOT = Path(__file__).resolve().parents[1]


class ExperimentType(Enum):
    """Type of data association experiment to run."""

    # Python port of the CLIPPER C++ benchmark over outlier ratios / # associations.
    CLIPPER_BENCHMARK = "CLIPPER_BENCHMARK"
    # Adversarial data association experiment (to be described later).
    ADVERSARIAL = "ADVERSARIAL"


@dataclass
class BunnyExperimentConfig:
    # Name of the experiment, used for saving results.
    experiment_name: str = "default_experiment"
    # Which experiment to run.
    experiment_type: ExperimentType = ExperimentType.CLIPPER_BENCHMARK
    # Path (relative to ROOT) to the .ply point cloud used to generate problems.
    ply_path: Path = Path("data/bun10k.ply")
    # Side length of the cube the point cloud is rescaled into.
    scale_cube_size: float = 1.0
    # Seed for reproducibility.
    seed: int = 0
    # Save results
    save_results: bool = True

    # --- Data association config (CLIPPER invariant parameters, etc.) ---
    data_association_config: DataAssociationConfig = field(
        default_factory=lambda: DataAssociationConfig(
            method=DataAssociationMethod.CLIPPER,
            invariant_sigma=0.015,
            invariant_epsilon=0.05,
        )
    )

    # --- Synthetic noise parameters (bounded normal noise) ---
    noise_sigma: float = 0.01
    noise_beta: float = 5.54 * 0.01

    # ---  sweep parameters ---
    # Numbers of putative associations to sweep over.
    num_assocs: List[int] = field(default_factory=lambda: [64, 256, 512, 1024, 2048])
    # Outlier ratios to sweep over, in [0, 1].
    outlier_ratios: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.8, 0.9]
    )
    # Number of Monte Carlo trials per (rho, m) configuration.
    num_trials: int = 20


def load_experiment_config(config_path: Path) -> BunnyExperimentConfig:
    # Start with defaults from the dataclass.
    config = OmegaConf.structured(BunnyExperimentConfig)
    # Merge overrides if provided.
    if config_path:
        overrides = OmegaConf.load(ROOT / config_path)
        config = OmegaConf.merge(config, overrides)
    return OmegaConf.to_object(config)


# ----------------------------------------------------------------------------
# Point cloud / synthetic correspondence helpers (Python ports of bm_utils.cpp)
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


def generate_bounded_normal_noise(
    n: int, sigma: float, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate ``n`` 3-vectors of normal noise, each with norm bounded by ``beta``."""
    eta = np.zeros((n, 3))
    for i in range(n):
        while True:
            v = rng.normal(0.0, sigma, size=3)
            if np.linalg.norm(v) <= beta:
                break
        eta[i] = v
    return eta


def make_noisy(pcd0: np.ndarray, sigma: float, beta: float, rng: np.random.Generator):
    """Return a copy of ``pcd0`` perturbed by bounded normal noise."""
    return pcd0 + generate_bounded_normal_noise(pcd0.shape[0], sigma, beta, rng)


def distance_based_correspondences(
    pcd0: np.ndarray,
    pcd1: np.ndarray,
    knn: int,
    radius: float,
    enforce_1to1: bool,
) -> np.ndarray:
    """Nearest-neighbour correspondences between two point clouds within ``radius``.

    Returns an (M, 2) integer array of (index in pcd0, index in pcd1) pairs.
    """
    # Define a KDTree to find the nearest neighbours of pcd0 in pcd1.
    tree = KDTree(pcd1)
    dists, idxs = tree.query(pcd0, k=knn)
    # Normalise to (N, knn) so knn == 1 and knn > 1 are handled uniformly.
    dists = np.atleast_2d(dists.T).T if knn > 1 else dists[:, None]
    idxs = np.atleast_2d(idxs.T).T if knn > 1 else idxs[:, None]

    # Map each pcd1 point to the candidate pcd0 points that matched it (with dist).
    corres: dict[int, list[tuple[int, float]]] = {}
    pairs: list[tuple[int, int]] = []
    for c0 in range(pcd0.shape[0]):
        for j in range(knn):
            c1 = int(idxs[c0, j])
            d = float(dists[c0, j])
            if d <= radius:
                pairs.append((c0, c1))
                if enforce_1to1:
                    corres.setdefault(c1, []).append((c0, d))

    if not enforce_1to1:
        return np.array(pairs, dtype=np.int64).reshape(-1, 2)

    # For each pcd1 point keep only the closest pcd0 point (mutual 1-to-1).
    A = []
    for c1 in sorted(corres.keys()):
        c0_best, _ = min(corres[c1], key=lambda t: t[1])
        A.append((c0_best, c1))
    return np.array(A, dtype=np.int64).reshape(-1, 2)


def generate_synthetic_correspondences(
    pcd0: np.ndarray,
    pcd1: np.ndarray,
    Agood: np.ndarray,
    m: int,
    rho: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ``m`` putative associations with outlier ratio ``rho``.

    Returns ``(A, Agt)`` where ``A`` is the (m, 2) putative association set used in
    the experiment and ``Agt`` is the (ni, 2) subset of ground-truth inliers.
    """
    assert 0.0 <= rho <= 1.0, "outlier ratio must be in [0, 1]"

    ni = int(round(m * (1.0 - rho)))  # number of inliers in final set
    no = m - ni  # number of outliers in final set
    p = Agood.shape[0]  # number of good associations to draw from
    if ni > p:
        raise ValueError(
            f"Not enough initial inliers ({p}) for the requested outlier ratio "
            f"({rho}, need {ni})."
        )

    A = np.zeros((m, 2), dtype=np.int64)
    Agt = np.zeros((ni, 2), dtype=np.int64)

    # Choose ni good associations without replacement.
    I = rng.permutation(p)
    for i in range(ni):
        Agt[i] = Agood[I[i]]
        A[no + i] = Agood[I[i]]

    # Sample outlier associations that are not part of the good set.
    good_set = set(map(tuple, Agood.tolist()))
    n0, n1 = pcd0.shape[0], pcd1.shape[0]
    tried: set = set()
    nele = 0
    while nele < no:
        a = int(rng.integers(0, n0))
        b = int(rng.integers(0, n1))
        if (a, b) in tried:
            continue
        tried.add((a, b))
        if (a, b) in good_set:
            continue
        A[nele] = (a, b)
        nele += 1

    return A, Agt


def get_precision_recall(A: np.ndarray, Agt: np.ndarray) -> Tuple[float, float]:
    """Precision and recall of selected associations ``A`` against ground truth ``Agt``."""
    if Agt.shape[0] == 0 or A.shape[0] == 0:
        return 0.0, 0.0
    gt_set = set(map(tuple, Agt.tolist()))
    TP = sum(1 for row in A.tolist() if tuple(row) in gt_set)
    precision = TP / A.shape[0]
    recall = TP / Agt.shape[0]
    return precision, recall


# ----------------------------------------------------------------------------
# Experiment drivers
# ----------------------------------------------------------------------------


def _make_clipper_block(cfg: BunnyExperimentConfig) -> DataAssociationBlock:
    return DataAssociationBlock(cfg.data_association_config)


def _associations_to_keypoints(
    pcd0: np.ndarray, pcd1: np.ndarray, A: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build homogeneous (4, m) keypoint tensors from an association set.

    The DataAssociationBlock treats keypoints sharing an index as putative
    correspondences, so we gather source/target points along the rows of ``A``.
    """
    src = pcd0[A[:, 0]]  # (m, 3)
    trg = pcd1[A[:, 1]]  # (m, 3)

    def to_homog(pts: np.ndarray) -> torch.Tensor:
        h = np.ones((pts.shape[0], 4), dtype=np.float64)
        h[:, :3] = pts
        return torch.from_numpy(h.T).float()  # (4, m)

    return to_homog(src), to_homog(trg)


def run_clipper_benchmark(cfg: BunnyExperimentConfig) -> pd.DataFrame:
    """Python port of the CLIPPER C++ benchmark using the DataAssociationBlocks."""
    rng = np.random.default_rng(cfg.seed)

    pcd0 = read_ply(ROOT / cfg.ply_path)
    pcd0 = scale_to_cube(pcd0, cfg.scale_cube_size)

    clipper_block = _make_clipper_block(cfg)
    
    # generate random (R,t)
    T_21 = np.eye(4)
    T_21[0:3,0:3] = Rotation.random().as_matrix()
    T_21[0:3,3] = np.random.uniform(low=-5, high=5, size=(3,))

    output_data = []
    total = len(cfg.outlier_ratios) * len(cfg.num_assocs) * cfg.num_trials
    with tqdm(total=total, desc="CLIPPER benchmark") as pbar:
        for rho in cfg.outlier_ratios:
            for m in cfg.num_assocs:
                for trial in range(cfg.num_trials):
                    # Noisy copy of the bunny and its ground-truth associations.
                    pcd1 = make_noisy(pcd0, cfg.noise_sigma, cfg.noise_beta, rng)
                    Agt0 = distance_based_correspondences(
                        pcd0, pcd1, knn=1, radius=cfg.noise_beta, enforce_1to1=True
                    )

                    # Synthetic putative correspondences with outlier ratio rho.
                    A, Agt = generate_synthetic_correspondences(
                        pcd0, pcd1, Agt0, m, rho, rng
                    )
                    # Apply a random transformation to the points
                    
                    
                    src_t, trg_t = _associations_to_keypoints(pcd0, pcd1, A)

                    # Time affinity-matrix construction.
                    t1 = time.perf_counter()
                    clipper_block.set_up_affinity_matrix(src_t, trg_t)
                    t2 = time.perf_counter()
                    t_affinity = t2 - t1

                    # Time the dense clique solver.
                    t1 = time.perf_counter()
                    inliers, _ = clipper_block.run_clipper()
                    t2 = time.perf_counter()
                    t_solver = t2 - t1

                    # Precision / recall of the selected associations.
                    Ain = A[inliers.cpu().numpy()]
                    precision, recall = get_precision_recall(Ain, Agt)

                    output_data.append(
                        dict(
                            rho=rho,
                            m=m,
                            trial=trial,
                            t_affinity=t_affinity,
                            t_solver=t_solver,
                            precision=precision,
                            recall=recall,
                            num_inliers=int(inliers.sum().item()),
                        )
                    )
                    pbar.update(1)

    df = pd.DataFrame(output_data)

    # Print an aggregated summary table (mean +/- std) mirroring the C++ benchmark.
    summary = (
        df.groupby(["rho", "m"])
        .agg(
            t_affinity_ms_mean=("t_affinity", lambda s: s.mean() * 1e3),
            t_affinity_ms_std=("t_affinity", lambda s: s.std() * 1e3),
            t_solver_ms_mean=("t_solver", lambda s: s.mean() * 1e3),
            t_solver_ms_std=("t_solver", lambda s: s.std() * 1e3),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
        )
        .reset_index()
    )
    print("\n" + summary.to_string(index=False))

    return df


def run_adversarial(cfg: BunnyExperimentConfig) -> pd.DataFrame:
    """Adversarial data association experiment.

    Scaffolding only -- the adversarial setup will be described later. This
    should construct point clouds / correspondences designed to be difficult for
    the DataAssociationBlock (e.g. structured outliers, near-symmetries), run the
    block, and record how the association and its certificate behave.
    """
    raise NotImplementedError(
        "The adversarial experiment has not been implemented yet."
    )


def run_experiment(cfg: BunnyExperimentConfig):
    # Seed everything for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.experiment_type == ExperimentType.CLIPPER_BENCHMARK:
        df = run_clipper_benchmark(cfg)
    elif cfg.experiment_type == ExperimentType.ADVERSARIAL:
        df = run_adversarial(cfg)
    else:
        raise ValueError(f"Unknown experiment type: {cfg.experiment_type}")

    if cfg.save_results:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M")
        run_dir = ROOT / "results" / "data_association" / cfg.experiment_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(run_dir / "results.csv", index=False)
        OmegaConf.save(OmegaConf.structured(cfg), run_dir / "experiment.yaml")
        print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="benchmark_test.yaml")
    args = parser.parse_args()

    exp_cfg_path = ROOT / "configs" / "data_association_experiments" / args.filename
    exp_config = load_experiment_config(exp_cfg_path)
    run_experiment(exp_config)
