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
from typing import List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.spatial import KDTree
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation
from pylgmath import Transformation

from stereo_loc.DataAssociationBlocks import (
    DataAssociationBlock,
    DataAssociationConfig,
    DataAssociationMethod,
)
from stereo_loc.PointCloudRegistrationBlock import estimate_pose_svd

ROOT = Path(__file__).resolve().parents[2]


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
    # Registration results
    registration: bool = False
    # Plot the final problem instance (point clouds + associations) at the end.
    plot: bool = False
    # --- Data association config (CLIPPER invariant parameters, etc.) ---
    data_association_config: DataAssociationConfig = field(
        default_factory=lambda: DataAssociationConfig(
            method=DataAssociationMethod.CLIPPER,
            invariant_sigma=0.015,
            invariant_epsilon=0.05,
        )
    )
    # Data association methods to sweep over for each problem instance.
    methods: List[DataAssociationMethod] = field(
        default_factory=lambda: [DataAssociationMethod.CLIPPER]
    )

    # --- Poor initialization for CLIPPER ---
    # If True, initialize CLIPPER with a poor initial solution (all outliers).
    poor_init: bool = False
    # Option to also retrieve the global solution by solving the SDP
    get_global_solution: bool = False
    # --- Synthetic noise parameters (bounded normal noise) ---
    noise_sigma: float = 0.01
    noise_beta: float = 5.54 * 0.01
    # --- Synthetic transformation parameters (random rotation + translation) ---
    # Ground truth transformation applied to the bunny to generate the target point cloud.
    # If None, random translation in [-1.0, 1.0]^3
    translation: List[float] | None = None
    # If None, random rotation
    rotation: List[float] | None = None
    # Adversarial transformation applied to the bunny to generate the target point cloud.
    # If None, random translation in [-1.0, 1.0]^3
    translation_adv: List[List[float]] | None = None
    # If None, random rotation
    rotation_adv: List[List[float]] | None = None
    # RANSAC multiplier for threshold (invariant_sigma * ransac_thresh_multiplier)
    ransac_thresh_multiplier: float = 0.5

    # ---  sweep parameters ---
    # Numbers of putative associations to sweep over.
    num_assocs: List[int] = field(default_factory=lambda: [64, 256, 512, 1024, 2048])
    # Outlier ratios to sweep over, in [0, 1].
    outlier_ratios: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.8, 0.9]
    )
    # Number of Monte Carlo trials per (rho, m) configuration.
    num_trials: int = 20
    # Sweep bounds for the noise thresholds.
    invariant_mult_min: float = 1.0
    invariant_mult_max: float = 1.0
    # Number of values for the threshold multiplier to sweep over (log spaced between min and max).
    invariant_mult_num: int = 1


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


def run_data_association(
    block: DataAssociationBlock,
    method: DataAssociationMethod,
    kpt_3D_src: torch.Tensor,
    kpt_3D_trg: torch.Tensor,
    x_init: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, np.ndarray | torch.Tensor]:
    """Dispatch to the configured data association solver.

    The affinity matrix is assumed to already have been set up (via
    :meth:`DataAssociationBlock.set_up_affinity_matrix`) so that its construction
    can be timed separately. CLIPPER, PMC and SDP reuse that cached matrix; RANSAC
    needs the raw keypoints to estimate poses and rebuilds the affinity internally.

    Returns ``(inliers, soln)`` where ``soln`` is the solution vector used for
    certification.
    """
    try:
        if method == DataAssociationMethod.CLIPPER:
            inliers, soln = block.run_clipper(x_init=x_init)
        elif method == DataAssociationMethod.PMC:
            inliers, soln, _ = block.run_pmc()
        elif method == DataAssociationMethod.SDP:
            inliers, soln = block.run_sdp()
        elif method == DataAssociationMethod.RANSAC:
            inliers, soln, _ = block.run_ransac(kpt_3D_src, kpt_3D_trg)
        else:
            raise ValueError(f"Invalid data association method: {method}")
    except Exception as e:
        print(f"Data association solver failed with exception: {e}")
        inliers = torch.zeros(block.num_assocs, dtype=torch.bool)
        soln = None
    return inliers, soln


def get_transformation(
    rotation: Optional[np.ndarray],
    translation: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a random homogeneous transform ``T_10``.

    Uses ``rotation`` (a rotation vector) and ``translation`` when provided,
    otherwise samples a random axis (with fixed magnitude) for each.
    """
    # Fixed magnitudes used when the rotation/translation are sampled randomly.
    ROTATION_ANGLE = np.pi / 4  # radians
    TRANSLATION_NORM = 1.0

    if rotation is not None:
        R_10 = Rotation.from_rotvec(np.array(rotation)).as_matrix()
    else:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R_10 = Rotation.from_rotvec(ROTATION_ANGLE * axis).as_matrix()
    if translation is not None:
        t_01_1 = np.array(translation)
    else:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        t_01_1 = TRANSLATION_NORM * axis
    T_10 = np.eye(4)
    T_10[:3, :3] = R_10
    T_10[:3, 3] = t_01_1
    return torch.from_numpy(T_10).float()


def clipper_benchmark_setup(
    cfg: BunnyExperimentConfig,
    pcd0: np.ndarray,
    m: int,
    rho: float,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Generate a single CLIPPER-benchmark problem instance.

    Builds a noisy, randomly transformed copy of the bunny together with ``m``
    putative correspondences at outlier ratio ``rho`` (a Python port of the
    CLIPPER C++ benchmark).

    Returns ``(src_t, trg_t, A, Agt)`` where ``src_t``/``trg_t`` are the
    homogeneous keypoint tensors fed to the solver, ``A`` is the (m, 2) putative
    association set and ``Agt`` is the ground-truth inlier subset.
    """
    # generate random (R,t)
    T_10 = get_transformation(cfg.rotation, cfg.translation, rng)

    # Noisy copy of the bunny and its ground-truth associations.
    pcd1 = make_noisy(pcd0, cfg.noise_sigma, cfg.noise_beta, rng)
    Agt0 = distance_based_correspondences(
        pcd0, pcd1, knn=1, radius=cfg.noise_beta, enforce_1to1=True
    )
    # Synthetic putative correspondences with outlier ratio rho.
    A, Agt = generate_synthetic_correspondences(pcd0, pcd1, Agt0, m, rho, rng)
    # map into keypoints
    src, trg_aligned = _associations_to_keypoints(pcd0, pcd1, A)
    # Apply transformation to the target keypoints
    trg = T_10 @ trg_aligned
    return src, trg, A, Agt, T_10


def adversarial_setup(
    cfg: BunnyExperimentConfig,
    pcd0: np.ndarray,
    m: int,
    rho: float,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single adversarial problem instance.

    Builds a noisy, randomly transformed copy of the bunny together with ``m``
    putative correspondences at outlier ratio ``rho``. The outliers are split
    (roughly evenly) across one or more *adversarial* point clouds, each with its
    own transformation: the number of adversarial clouds is given by
    ``len(cfg.translation_adv)`` (defaulting to a single random adversarial
    transform when ``translation_adv``/``rotation_adv`` are unset).

    Returns ``(src_t, trg_t, A, Agt, T_10)`` where ``src_t``/``trg_t`` are the
    homogeneous keypoint tensors fed to the solver, ``A`` is the (m, 2) putative
    association set, ``Agt`` is the ground-truth inlier subset and ``T_10`` is the
    true transformation.
    """
    # Generate true transformation (R,t) for the bunny.
    T_10 = get_transformation(cfg.rotation, cfg.translation, rng)
    # Generate an adversarial transformation (R,t) per adversarial point cloud.
    if cfg.translation_adv is not None:
        num_adv = len(cfg.translation_adv)
    elif cfg.rotation_adv is not None:
        num_adv = len(cfg.rotation_adv)
    else:
        num_adv = 1
    T_10_adv = [
        get_transformation(
            cfg.rotation_adv[i] if cfg.rotation_adv is not None else None,
            cfg.translation_adv[i] if cfg.translation_adv is not None else None,
            rng,
        )
        for i in range(num_adv)
    ]

    # Noisy copy of the bunny and its ground-truth associations.
    pcd1 = make_noisy(pcd0, cfg.noise_sigma, cfg.noise_beta, rng)
    Agt0 = distance_based_correspondences(
        pcd0, pcd1, knn=1, radius=cfg.noise_beta, enforce_1to1=True
    )
    # Number of correspondences to draw from the good set, and adversarial set.
    ni = int(round(m * (1.0 - rho)))  # number of inliers in final set
    no = m - ni  # number of outliers in final set

    # Synthetic putative correspondences with outlier ratio rho.
    rng = set_seed(cfg.seed)
    Ai, Agt = generate_synthetic_correspondences(pcd0, pcd1, Agt0, ni, 0.0, rng)
    rng = set_seed(cfg.seed)
    Ao, _ = generate_synthetic_correspondences(pcd0, pcd1, Agt0, no, 0.0, rng)
    A = np.vstack((Ao, Ai))
    # map into keypoints
    src, trg_aligned = _associations_to_keypoints(pcd0, pcd1, A)
    # Apply each adversarial transformation to its share of the outliers.
    outlier_kpts = trg_aligned[:, : Ao.shape[0]]
    outlier_splits = np.array_split(np.arange(Ao.shape[0]), num_adv)
    trg_outliers = [
        T_10_adv[i] @ outlier_kpts[:, split] for i, split in enumerate(outlier_splits)
    ]
    # Apply the true transformation to the inliers.
    trg_inliers = T_10 @ trg_aligned[:, Ao.shape[0] :]
    trg = torch.cat((*trg_outliers, trg_inliers), dim=1)

    return src, trg, A, Agt, T_10


def plot_associations(
    src_t: torch.Tensor,
    trg_t: torch.Tensor,
    inliers: torch.Tensor,
    inliers_global: torch.Tensor | None = None,
    num_outliers: int = 0,
    certified: bool = False,
) -> None:
    """Plot the source/target point clouds and their putative associations.

    Source keypoints are drawn in red, target keypoints in blue. Each association
    is drawn as a line connecting the corresponding source and target point:
    green for the inliers selected by the data association solver and red for the
    remaining (outlier) associations. Association lines are drawn with an alpha of
    0.5.
    """
    import matplotlib.pyplot as plt

    # (m, 3) source/target points for each association (drop homogeneous row).
    src = src_t[:3, :].cpu().numpy().T
    trg = trg_t[:3, :].cpu().numpy().T

    # Inlier mask as selected by the data association solver.
    is_inlier = inliers.cpu().numpy().astype(bool)
    is_inlier_global = (
        inliers_global.cpu().numpy().astype(bool)
        if inliers_global is not None
        else None
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Fill the figure: remove the default subplot padding and let the 3D axes
    # occupy the entire figure area.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Point clouds: source (red) and target (blue).
    ax.scatter(src[:, 0], src[:, 1], src[:, 2], c="magenta", s=5, label="Source")
    if num_outliers > 0:
        ax.scatter(
            trg[:num_outliers, 0],
            trg[:num_outliers, 1],
            trg[:num_outliers, 2],
            c="red",
            s=5,
            label="Target (outliers)",
        )
    ax.scatter(
        trg[num_outliers:, 0],
        trg[num_outliers:, 1],
        trg[num_outliers:, 2],
        c="blue",
        s=5,
        label="Target (inliers)",
    )
    # Association lines: green for inliers, red for outliers, alpha 0.5. When a
    # global solution is available, disagreements are highlighted: red where the
    # local solution flags an inlier the global one rejects, orange for the
    # reverse, and lines the global solution also rejects are not drawn.
    for i in range(src.shape[0]):
        lw = 0.5
        alpha = 0.3
        if inliers_global is not None:
            if is_inlier[i] and is_inlier_global[i]:
                color = "green"
            elif is_inlier[i] and not is_inlier_global[i]:
                color = "red"
                lw = 2.0
                alpha = 1.0
            elif not is_inlier[i] and is_inlier_global[i]:
                color = "orange"
                lw = 2.0
                alpha = 1.0
            else:
                continue
        else:
            color = "green" if is_inlier[i] else "red"
        ax.plot(
            [src[i, 0], trg[i, 0]],
            [src[i, 1], trg[i, 1]],
            [src[i, 2], trg[i, 2]],
            color=color,
            alpha=alpha,
            linewidth=lw,
        )
    ax.set_title(f"Data Association (certified={certified})")
    ax.legend()
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # Zoom in so the point cloud fills the axes (reduces the whitespace that
    # matplotlib leaves around 3D data).
    ax.set_aspect("equal")
    ax.margins(0)
    ax.set_position([0, 0, 1, 1])
    # No Background
    ax.grid(False)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.axis("off")
    # Set camera to look down the negative Z axis, with Y up
    ax.view_init(elev=90, azim=-90)

    return ax


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    return rng


def run_experiment(cfg: BunnyExperimentConfig):
    rng = set_seed(cfg.seed)
    # Seed everything for reproducibility.
    pcd0 = read_ply(ROOT / cfg.ply_path)
    pcd0 = scale_to_cube(pcd0, cfg.scale_cube_size)
    # center the points on zero
    pcd0 -= pcd0.mean(axis=0)

    data_association = DataAssociationBlock(cfg.data_association_config)

    # Select the problem-generation function based on the experiment type. The
    # two experiments only differ in how each problem instance is constructed.
    if cfg.experiment_type == ExperimentType.CLIPPER_BENCHMARK:
        setup_fn = clipper_benchmark_setup
    elif cfg.experiment_type == ExperimentType.ADVERSARIAL:
        setup_fn = adversarial_setup
    else:
        raise ValueError(f"Unknown experiment type: {cfg.experiment_type}")

    # Generate array of multiplier values for the invariant values
    invariant_mults = np.logspace(
        np.log10(cfg.invariant_mult_min),
        np.log10(cfg.invariant_mult_max),
        cfg.invariant_mult_num,
    )

    # Run main loop
    output_data = []
    total = (
        len(cfg.outlier_ratios)
        * len(cfg.num_assocs)
        * cfg.num_trials
        * len(invariant_mults)
    )
    index = 0
    with tqdm(total=total, desc=cfg.experiment_type.value) as pbar:
        for invariant_mult in invariant_mults:
            # Reset Clipper using the new invariant values for this multiplier
            invariant_sigma = (
                cfg.data_association_config.invariant_sigma * invariant_mult
            )
            invariant_epsilon = (
                cfg.data_association_config.invariant_epsilon * invariant_mult
            )
            data_association.set_clipper(
                invariant_sigma=invariant_sigma, invariant_epsilon=invariant_epsilon
            )
            # Set RANSAC threshold to invariant_sigma
            # Note: threshold should be at or below 0.5 to return inliers
            data_association.config.ransac_inlier_threshold = float(
                invariant_epsilon * cfg.ransac_thresh_multiplier
            )
            for rho in cfg.outlier_ratios:
                for m in cfg.num_assocs:
                    for trial in range(cfg.num_trials):
                        # Reset rng for reproducibility across trials.
                        index += 1
                        rng = set_seed(cfg.seed + index)
                        # Generate the problem instance for this experiment.
                        src_t, trg_t, A, Agt, T_trg_src_gt_np = setup_fn(
                            cfg, pcd0, m, rho, rng
                        )

                        # Time affinity-matrix construction.
                        t1 = time.perf_counter()
                        data_association.set_up_affinity_matrix(src_t, trg_t)
                        t2 = time.perf_counter()
                        t_affinity = t2 - t1
                        # Initialization for CLIPPER
                        if cfg.poor_init:
                            n_outlier = len(A) - len(Agt)
                            x_init = np.zeros(src_t.shape[1], dtype=np.float64)
                            x_init[:n_outlier] = 1.0
                        else:
                            x_init = None
                        # Run each configured solver (CLIPPER, PMC, SDP or RANSAC) on
                        # the same problem instance, reusing the cached affinity matrix.
                        inliers_list = []
                        for method in cfg.methods:
                            # Reset rng seed again (due to inconsistency in solver rng calls.)
                            rng = set_seed(cfg.seed + trial)
                            t1 = time.perf_counter()
                            inliers, soln = run_data_association(
                                data_association, method, src_t, trg_t, x_init=x_init
                            )
                            t2 = time.perf_counter()
                            t_solver = t2 - t1
                            # if enabled, also get the globally optimal solution by solving the SDP
                            inliers_global, soln_global = None, None
                            if cfg.get_global_solution:
                                inliers_global, soln_global = data_association.run_sdp()

                            # Optionally certify the data association solution and time it.
                            data_association_certified = False
                            t_certify = np.nan
                            num_iter_cert = None
                            if cfg.data_association_config.certify and soln is not None:
                                cert_result_da = data_association.certify_solution(soln)
                                t_certify = cert_result_da.solver_time
                                num_iter_cert = cert_result_da.num_iterations
                                data_association_certified = cert_result_da.certified

                            # Precision / recall of the selected associations.
                            if inliers is not None:
                                Ain = A[inliers.cpu().numpy()]
                                precision, recall = get_precision_recall(Ain, Agt)
                            else:
                                precision, recall = None, None

                            # Registration
                            if (
                                cfg.registration
                                and inliers is not None
                                and inliers.sum() > 0
                            ):
                                # Restrict measurements to inliers and estimate the transformation using SVD
                                src_inliers = src_t[:, inliers]
                                trg_inliers = trg_t[:, inliers]
                                T = estimate_pose_svd(src_inliers, trg_inliers)
                                # Compute the relative error between the estimated transformation and the ground truth
                                T_trg_src = Transformation(T_ba=T.cpu().numpy())
                                T_trg_src_gt = Transformation(T_ba=T_trg_src_gt_np)
                                # Assume right perturbation: T_est = T_gt * T_error
                                T_error = T_trg_src_gt.inverse() * T_trg_src
                                # Lie algebra vector (se(3)) of the pose error.
                                xi_err = np.asarray(T_error.vec()).ravel()
                            else:
                                xi_err = np.full(6, np.nan)

                            # Store the results
                            output_data.append(
                                dict(
                                    method=method.value,
                                    outlier_ratio=rho,
                                    num_assoc=m,
                                    trial=trial,
                                    t_affinity=t_affinity,
                                    t_solver=t_solver,
                                    t_certify=t_certify,
                                    cert_da=data_association_certified,
                                    precision=precision,
                                    recall=recall,
                                    num_inliers=(
                                        int(inliers.sum().item())
                                        if inliers is not None
                                        else None
                                    ),
                                    obj_value=data_association.obj_value,
                                    num_constraints=data_association.num_constraints,
                                    num_iter_cert=num_iter_cert,
                                    **{
                                        f"xi_err_{i}": xi_err[i]
                                        for i in range(6)
                                    },
                                    inv_mult=invariant_mult,
                                )
                            )
                            inliers_list.append(inliers)
                        pbar.update(1)

    df = pd.DataFrame(output_data)

    if cfg.save_results:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M")
        run_dir = (
            ROOT / "results" / "data_association" / cfg.experiment_name / timestamp
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(run_dir / "results.csv", index=False)
        OmegaConf.save(OmegaConf.structured(cfg), run_dir / "experiment.yaml")
        print(f"\nSaved results to {run_dir}")
    else:
        print("\nExperiment results:")
        print(df)
        # print whether was certified or not
        print("\nCertification results:")
        print(df[["method", "cert_da", "t_certify", "num_iter_cert"]])

    # Plot the final problem instance (point clouds + associations).
    # if cfg.experiment_type == ExperimentType.ADVERSARIAL:
    #     num_outliers = len(A) - len(Agt)
    # else:
    #     num_outliers = 0
    num_outliers = 0
    if cfg.plot:
        ax = plot_associations(
            src_t,
            trg_t,
            inliers,
            inliers_global,
            num_outliers,
            data_association_certified,
        )
        if cfg.save_results:
            fig_path = run_dir / "associations.png"
            ax.get_figure().savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {fig_path}")
        else:
            from matplotlib import pyplot as plt

            plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="benchmark_test.yaml")
    args = parser.parse_args()

    exp_cfg_path = ROOT / "configs" / "data_association_experiments" / args.filename
    exp_config = load_experiment_config(exp_cfg_path)
    run_experiment(exp_config)
