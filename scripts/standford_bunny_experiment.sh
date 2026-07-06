# Go to experiments directory
cd /workspace/experiments
# activate env
. .venv/bin/activate

# Run sweep experiment (low assoc count for SDP comparison)
python scripts/standford_bunny_experiment.py benchmark_sweep_low.yaml
python scripts/standford_bunny_experiment.py benchmark_sweep_low_sdp.yaml
