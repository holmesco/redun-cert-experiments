# Go to experiments directory
cd /workspace/experiments
# activate env
. .venv/bin/activate

# Run parameter sweep experiments 
python src/experiments/stanford_bunny_experiments.py invariant_sweep.yaml

# Run timing performance experiments
# Sweep outlier ratios
python src/experiments/stanford_bunny_experiments.py timing_sweep_outlier_ratio.yaml
# Sweep associations
python src/experiments/stanford_bunny_experiments.py timing_sweep_num_assoc.yaml
