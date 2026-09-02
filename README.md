# CP-Cert Experiments

Experiment code for the paper

> *Following a Unique Path: A Fast Certifier Applied to Outlier-Robust Pose
> Registration*, draft, August 2026.

The certifier itself (**CP-Cert**) lives in the companion
[RankTools](../ranktools) repository, which provides the C++ implementation and
the `ranktools` Python bindings. This repository contains only the *experiments*
of Section V: the simulated Stanford-bunny data-association and pose-registration
studies, and the stereo pose-registration pipeline run on the EuRoC dataset.
Section, equation, figure, and table numbers below refer to the paper.

Three experiment runners are provided, each corresponding to one experiment
group in the paper:

| Runner | Paper section | What it does |
| --- | --- | --- |
| [src/experiments/stanford_bunny_experiments.py](src/experiments/stanford_bunny_experiments.py) | V-A1 – V-A3 | Synthetic data-association problems from the Stanford bunny. Runs CLIPPER / PMC / RANSAC / a direct Mosek SDP solve, certifies each candidate with CP-Cert applied to the (LT) relaxation. |
| [src/experiments/pose_reg_experiment.py](src/experiments/pose_reg_experiment.py) | V-A4 | Matrix-weighted pose registration on synthetic stereo point clouds. Solves locally with GTSAM, certifies with CP-Cert, compares against Mosek. |
| [src/experiments/stereo_pipeline_experiment.py](src/experiments/stereo_pipeline_experiment.py) | V-B | The full certifiable stereo pipeline (Figure 8) run over a EuRoC sequence: SuperPoint + LightGlue matching, inverse stereo model, certified data association, certified pose registration. |

## Repository layout

```
configs/          experiment YAML configs, one directory per runner
data/             input data: point clouds, EuRoC sequences, network weights (gitignored)
extern/           third-party dependencies and the MOSEK distribution (gitignored / submodules)
results/          experiment outputs, one directory per runner (gitignored)
scripts/          post-processing: turns results.csv files into the paper's figures and tables
src/experiments/  the three experiment runners
src/stereo_loc/   pipeline blocks: data association, registration, EuRoC loading/preprocessing
src/mat_weight_loc/, src/utils/   matrix-weighted localization problem and shared utilities
tests/            pytest suite
```

`data/` and `results/` are in [.gitignore](.gitignore) — they must be populated
locally (see below), and every experiment output stays local.

---

# 1. Setting up `extern/`

`extern/` holds every dependency that is built from source, plus the MOSEK
distribution. All of these are wired into the `uv` workspace in
[pyproject.toml](pyproject.toml) (`[tool.uv.workspace] members`), so they must be
present *before* `uv sync` is run.

## 1.1 Repositories

Three of them are registered as git submodules in [.gitmodules](.gitmodules):

```bash
git submodule update --init --recursive
```

| Path | Provides | Remote |
| --- | --- | --- |
| `extern/certifiable-tools` | `cert_tools` — SDP/relaxation utilities | `git@github.com:utiasASRL/certifiable-tools.git` |
| `extern/clipper` | `clipperpy` — fork of CLIPPER with the PMC bindings exposed | `git@github.com:holmesco/clipper.git` |
| `extern/mat_weight_certs` | `mwcerts` — matrix-weighted problem definitions and relaxations ([19]) | `git@github.com:holmesco/mat_weight_certs.git` |

The remaining workspace members are **not** submodules and have to be cloned by
hand into `extern/`:

```bash
cd extern
git clone git@github.com:utiasASRL/poly_matrix.git            # poly-matrix
git clone https://github.com/cvg/LightGlue.git                # lightglue     (feature matching)
git clone https://github.com/Jdiaz031/RAFT-Stereo.git         # raft_stereo   (disparity; fork with uv-compatible imports)
```

The RAFT-Stereo fork is required rather than the upstream repository: it fixes
the package imports so the module can be installed as `raft_stereo`.

CLIPPER pulls [PMC](https://github.com/jingnanshi/pmc) itself through CMake
`FetchContent` while building, so nothing has to be done for the PMC baseline. A
manual clone at `extern/pmc` is only useful for local debugging of that
dependency.

After this step `extern/` should contain:

```
extern/
├── certifiable-tools/          (submodule)
├── clipper/                    (submodule)
├── mat_weight_certs/           (submodule)
├── poly_matrix/                (clone)
├── LightGlue/                  (clone)
├── RAFT-Stereo/                (clone)
├── mosektoolslinux64x86.tar.bz2
└── mosek.lic
```

## 1.2 MOSEK

Mosek is the interior-point baseline that CP-Cert is compared against
throughout Section V, and it is also how rank tightness is checked. Place the
Linux distribution and your license file in `extern/`:

* `extern/mosektoolslinux64x86.tar.bz2` — MOSEK 11.1 for `linux64x86`, from
  [mosek.com](https://www.mosek.com/downloads/).
* `extern/mosek.lic` — your license file (academic licenses are free).

Install it and build the Fusion C++ API (this is what RankTools links against):

```bash
sudo tar -xjf extern/mosektoolslinux64x86.tar.bz2 -C /opt
cd /opt/mosek/11.1/tools/platform/linux64x86/src/fusion_cxx && make install
```

then export:

```bash
export MOSEK_HOME=/opt/mosek/11.1/tools/platform/linux64x86
export PATH="$MOSEK_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MOSEK_HOME/bin:$LD_LIBRARY_PATH"
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
```

`/opt/mosek/11.1/tools/platform/linux64x86` is the location RankTools' CMake
expects (`MOSEK_DIR` in its `CMakeLists.txt`); adjust there if you install
elsewhere. The [Docker/Dockerfile](../ranktools/Docker/Dockerfile) in RankTools
performs exactly these steps and is the easiest way to reproduce the environment.

## 1.3 Python environment

The experiments depend on RankTools by **path**
(`ranktools = { path = "../ranktools" }`), so the two repositories must sit
side by side:

```
<workspace>/
├── ranktools/
└── experiments/
```

Install the system libraries needed by `scikit-sparse`, then sync:

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libsuitesparse-dev libopenblas-dev gfortran

cd experiments
uv sync                      # builds ranktools + clipperpy from source, installs CUDA 12.1 torch wheels
source .venv/bin/activate
```

`uv sync` creates `.venv/` here and installs `src/` as an editable package, which
is what makes `stereo_loc`, `utils`, and `mat_weight_loc` importable when the
runners are launched as scripts. If you later add a module by hand
(`uv pip install ...`), use `uv sync --inexact` afterwards so it is not removed.

Check the install with:

```bash
pytest
```

---

# 2. Setting up `data/`

All input data lives under `data/` and is referenced from the configs by paths
**relative to the repository root**. The expected layout is:

```
data/
├── bun10k.ply                              Stanford bunny (10k points)
├── Armadillo.ply                           optional, unused by the paper configs
├── raft_stereo/
│   └── raftstereo-middlebury.pth           RAFT-Stereo checkpoint
└── Euroc/
    ├── EurocStereo.yaml                    rectification + stereo calibration
    └── MH_01_easy/
        └── mav0/
            ├── cam0/                       left images  + data.csv, sensor.yaml
            ├── cam1/                       right images + data.csv, sensor.yaml
            ├── state_groundtruth_estimate0/ ground-truth trajectory
            └── disparities/                generated in the preprocessing step below
```

**Stanford bunny.** `data/bun10k.ply` is used by both simulated experiments
(`ply_path` in their configs). Obtain it from the
[Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)
and subsample/convert to a 10k-point `.ply`. The runners rescale the cloud into a
cube of side `scale_cube_size` and zero-center it, so absolute scale of the source
model does not matter.

**EuRoC.** Download the *Machine Hall 01 (easy)* sequence in ASL format from the
[EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
and unzip it so that `data/Euroc/MH_01_easy/mav0/` exists. Other sequences work
too — just point `dataset_path` in the experiment config at them.

**Stereo calibration.** `data/Euroc/EurocStereo.yaml` holds the rectification
maps and stereo intrinsics; the paper uses the same rectification and stereo
parameters as ORB-SLAM3 ([59]), i.e. its `EuRoC.yaml` calibration file. Note that
[EurocPreprocess](src/stereo_loc/EurocPreprocess.py) defaults this path to the
absolute location `/workspace/experiments/data/Euroc/EurocStereo.yaml`; if the
repository is checked out elsewhere, pass `stereo_params=` explicitly or edit
that default.

**RAFT-Stereo weights.** The pipeline generates disparity maps with RAFT-Stereo
([58]). Fetch the checkpoints with the script that ships with the cloned repo and
place the Middlebury model where the preprocessing step expects it:

```bash
cd extern/RAFT-Stereo && bash download_models.sh && cd ../..
mkdir -p data/raft_stereo
cp extern/RAFT-Stereo/models/raftstereo-middlebury.pth data/raft_stereo/
```

**Disparity preprocessing (required before the stereo experiment).** The stereo
runner reads *precomputed* disparity images from `mav0/disparities/` — one 16-bit
PNG per left image, scaled by 256. Generate them once per sequence:

```python
from pathlib import Path
from stereo_loc.EurocPreprocess import EurocPreprocess

ds = EurocPreprocess(
    Path("data/Euroc/MH_01_easy"),
    raft_stereo_ckpt_path=Path("data/raft_stereo/raftstereo-middlebury.pth"),
)
ds.process_disparities(use_raft=True, device="cuda:0")
```

(`use_raft=False` falls back to OpenCV SGBM and writes to `disparities_sgbm/`
instead; the paper uses RAFT-Stereo.) The `__main__` block at the bottom of
[EurocPreprocess.py](src/stereo_loc/EurocPreprocess.py) contains the same call
along with helpers for inspecting and tuning disparities.

A CUDA GPU is effectively required for this step and strongly recommended for the
stereo experiment (SuperPoint/LightGlue/RAFT-Stereo); everything falls back to CPU
if `torch.cuda.is_available()` is false.

---

# 3. How the experiments work

Every experiment follows the same three-part contract: **a YAML config in
`configs/` → a runner in `src/experiments/` → a timestamped run directory in
`results/`.**

## 3.1 Config in, results out

Each runner takes a single positional argument: the *file name* of a YAML config,
which is resolved against that runner's own config directory (not against the
current working directory):

```bash
python src/experiments/stanford_bunny_experiments.py invariant_sweep.yaml
python src/experiments/pose_reg_experiment.py distance_sweep.yaml
python src/experiments/stereo_pipeline_experiment.py MH01e_clipper.yaml
```

| Runner | Reads configs from | Default config | Writes results to |
| --- | --- | --- | --- |
| `stanford_bunny_experiments.py` | [configs/data_association_experiments/](configs/data_association_experiments/) | `benchmark_test.yaml` | `results/data_association/` |
| `pose_reg_experiment.py` | [configs/pose_registration_experiments/](configs/pose_registration_experiments/) | `pose_reg_test.yaml` | `results/pose_registration/` |
| `stereo_pipeline_experiment.py` | [configs/stereo_experiments/](configs/stereo_experiments/) | `test.yaml` | `results/stereo_loc/` |

The config is loaded with [OmegaConf](https://omegaconf.readthedocs.io) as a
*structured* config: the runner's experiment dataclass supplies the defaults, and
the YAML file is merged on top of it. Only the fields you want to change need to
appear in the YAML; everything else falls back to the dataclass default, and a
misspelled key is an error rather than a silent no-op. The dataclasses are the
authoritative documentation of the available fields — see
`BunnyExperimentConfig`, `PoseRegExperimentConfig`, and
`StereoPipelineExperimentConfig` at the top of the respective runners, where every
field carries a comment.

The stereo runner adds one indirection: its own config selects the pipeline
config. [configs/stereo_pipeline/stereo_pipeline_default.yaml](configs/stereo_pipeline/stereo_pipeline_default.yaml)
holds the full pipeline settings (feature extractor, data association, pose
registration, and CP-Cert parameters for each), and the experiment's
`override_path` names a small override file — e.g.
[clipper_overrides.yaml](configs/stereo_pipeline/clipper_overrides.yaml),
`pmc_override.yaml`, `sdp_override.yaml`, `ransac_override.yaml` — which is merged
over those defaults to pick the data-association method and its parameters.

## 3.2 The results directory

When `save_results: True` (the default), a run writes to

```
results/<group>/<experiment_name>/<YYYYMMDDThhmm>/
```

where `<group>` is one of `data_association`, `pose_registration`, `stereo_loc`,
`<experiment_name>` is taken from the config, and the timestamp is the run's start
time — so re-running the same config never overwrites an earlier run. Each run
directory contains:

| File | Contents |
| --- | --- |
| `results.csv` | One row per trial: solver and certifier outcomes, costs, timings, iteration counts, errors. This is the only file the post-processing reads. |
| `experiment.yaml` | The *fully resolved* experiment config (defaults + overrides), so each run records exactly how it was produced. |
| `stereo_pipeline.yaml` | Stereo runs only: the resolved pipeline config, including all CP-Cert parameters. |
| `*.png` | Written when `plot: True` — e.g. `associations.png` / `affinity.png` (data association), `setup.png` / `setup_zoom.png` (pose registration), `figure.png` (stereo). |
| `frames.npz` | Pose registration only: initial, estimated, and ground-truth camera frames plus per-trial certification flags, used for the setup figure. |

Setting `save_results: False` skips the run directory entirely and prints the
results DataFrame to stdout — the intended mode for the `*_test.yaml` configs while
debugging.

## 3.3 Post-processing

The scripts in [scripts/](scripts/) turn run directories into the paper's figures
and tables. Each one discovers `results/<group>/<experiment_name>/<timestamp>/results.csv`
files, annotates each row with its `experiment_name` and `timestamp`, and
concatenates them — so several runs of the same config can be pooled, or a single
run selected. Passing `timestamp="latest"` picks the most recent run of an
experiment; passing an explicit timestamp string (e.g. `"20260805T2030"`) pins a
specific one.

| Script | Consumes | Produces |
| --- | --- | --- |
| [scripts/stanford_bunny_postproc.py](scripts/stanford_bunny_postproc.py) | `results/data_association/` | Score-parameter sweep figures (Figure 5) and the certifier confusion table (Table I); timing sweep figures (Figure 6). Figures are written to `results/data_association/{invariant_sweep,timing_sweep}/figures/`. |
| [scripts/pose_reg_postproc.py](scripts/pose_reg_postproc.py) | `results/pose_registration/` | Certification confusion and runtime tables by camera distance (Tables II and III), printed as text and LaTeX. |
| [scripts/stereo_pipeline_postproc.py](scripts/stereo_pipeline_postproc.py) | `results/stereo_loc/` | Machine Hall certification/error summary and certifier runtimes by frame interval (Tables IV and V), plus the invariant sweep plots. |

Run them directly (`python scripts/stanford_bunny_postproc.py`); the `__main__`
block at the bottom of each selects which analyses to run and which run timestamps
to use, and is the place to edit when pointing the analysis at your own runs.

Two other utilities are unrelated to the paper's results:
[scripts/conversion.py](scripts/conversion.py) and
[scripts/generate_hardcoded_sdp_cases.py](scripts/generate_hardcoded_sdp_cases.py)
export SDP problem instances into the text format read by the RankTools C++ tests.

---

# 4. Reproducing the paper

[scripts/run_all_experiments.sh](scripts/run_all_experiments.sh) runs the
configs behind the paper's results end to end. Individually:

```bash
# Data association (Section V-A1 – V-A3)
python src/experiments/stanford_bunny_experiments.py adv_min_good_init.yaml       # Fig. 1 (c): correct clique, certified
python src/experiments/stanford_bunny_experiments.py adv_min_poor_init.yaml       # Fig. 1 (d): adversarial local min, not certified
python src/experiments/stanford_bunny_experiments.py invariant_sweep.yaml         # Fig. 5 (a-c), Table I
python src/experiments/stanford_bunny_experiments.py inv_sweep_0p2.yaml           # Fig. 5 (d): too restrictive (alpha = 0.2)
python src/experiments/stanford_bunny_experiments.py inv_sweep_1.yaml             # Fig. 5 (e): ideal        (alpha = 1)
python src/experiments/stanford_bunny_experiments.py inv_sweep_30.yaml            # Fig. 5 (f): too permissive (alpha = 30)
python src/experiments/stanford_bunny_experiments.py timing_sweep_outlier_ratio.yaml  # Fig. 6 (b)
python src/experiments/stanford_bunny_experiments.py timing_sweep_num_assoc.yaml      # Fig. 6 (a)

# Pose registration (Section V-A4)
python src/experiments/pose_reg_experiment.py pose_reg_setup.yaml                 # Fig. 7
python src/experiments/pose_reg_experiment.py distance_sweep.yaml                 # Tables II, III

# Stereo pipeline on EuRoC MH01 (Section V-B)
python src/experiments/stereo_pipeline_experiment.py MH01e_clipper.yaml           # Tables IV, V
python src/experiments/stereo_pipeline_experiment.py test.yaml                    # Fig. 9: a single non-certified local minimum

# Figures and tables
python scripts/stanford_bunny_postproc.py
python scripts/pose_reg_postproc.py
python scripts/stereo_pipeline_postproc.py
```

The sweeps are long: `invariant_sweep.yaml` is 30 trials at each of 30 parameter
values across four methods (each including a direct Mosek solve), and
`MH01e_clipper.yaml` runs the full pipeline over every frame of MH01 at four frame
intervals. The `*_test.yaml` and `benchmark_test.yaml` configs are small,
single-trial versions for checking the setup first.

## 4.1 Certifier parameters

Every config exposes the CP-Cert parameters of Table VI through a nested
`cpcert_params` block (under `data_association_config` and/or
`registration_config`), which maps onto RankTools' `CPCertParams` via
[CPCertParamsConfig](src/stereo_loc/CPCertParamsConfig.py):

```yaml
cpcert_params:
  lin_solver: MFCG_LRP      # matrix-free PCG with the low-rank preconditioner (Sections III-D, III-E)
  lin_solve_max_iter: 200
  lin_solve_tol: 1e-8
  delta: 1e-7               # initial primal perturbation, X_o = X̂ + delta*I  (Eq. 16)
  max_iter: 10              # K_max
  early_stop_angle: True    # angle divergence test (Eq. 25)
  max_angle: 8e-5           # theta_max
  perturb_cost: True
  eps_cost: 1e-7            # epsilon_0, set equal to delta
  lrp_params:
    tau: 1e-7               # preconditioner perturbation, set equal to delta
    method: SparseLDLT
  adaptive_perturb: True    # adaptive epsilon update (Eq. 22)
  eps_mult_min: 1e-3        # epsilon_min / epsilon_0
```

The data-association and pose-registration columns of Table VI correspond to the
values in the data-association and pose-registration configs, respectively. See
the [RankTools README](../ranktools/README.md) for the full parameter reference
and the mapping from Algorithm 2 onto the API.

## 4.2 Notes

* **Method selection.** The data-association experiments sweep the `methods` list
  (`CLIPPER`, `PMC`, `RANSAC`, `SDP`); `SDP` is the direct Mosek solve used both as
  the timing baseline and as the ground truth for deciding whether a local solver
  reached the global optimum (relative cost gap below `1e-4`).
* **`spawn_process`.** For large problems the Mosek solve is run in a child process
  (`data_association_config.spawn_process: True`) so that an OOM kill takes down
  only the solve and not the whole sweep. Small sweeps can turn this off.
* **Plotting.** The stereo runner sets `DISPLAY=":32"` at import time for use in a
  headless container; change or remove that line in
  [stereo_pipeline_experiment.py](src/experiments/stereo_pipeline_experiment.py) if
  your setup differs. Plotting is only active when `plot: True`.
* **Reproducibility.** Every config carries a seed (`seed` / `random_seed`) that
  fixes the synthetic problem generation and the dataset sampling; the resolved
  config is stored alongside each set of results.
