# Go to experiments directory
cd /workspace/experiments
# activate env
. .venv/bin/activate

# STANFORD BUNNY EXPERIMENTS
# ==========================
# Run parameter sweep experiments 
python src/experiments/stanford_bunny_experiments.py invariant_sweep.yaml

# Run timing performance experiments
# Sweep outlier ratios
python src/experiments/stanford_bunny_experiments.py timing_sweep_outlier_ratio.yaml
# Sweep associations
python src/experiments/stanford_bunny_experiments.py timing_sweep_num_assoc.yaml

# POSE REGISTRATION EXPERIMENTS
# =============================
# Make Setup plots
python src/experiments/pose_reg_experiment.py pose_reg_setup.yaml
# Run sweep over different distances
python src/experiments/pose_reg_experiment.py distance_sweep.yaml

# STEREO PIPELINE EXPERIMENTS
# ===========================
python src/experiments/stereo_pipeline_experiment.py MH01e_clipper.yaml