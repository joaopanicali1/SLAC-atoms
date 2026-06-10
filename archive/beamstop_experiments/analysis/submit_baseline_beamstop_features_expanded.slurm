#!/bin/bash
#SBATCH --job-name=bs_feat_nsk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-pascal
#SBATCH --gres=gpu:1
#SBATCH --output=analysis/logs/baseline_features_expanded_%j.out
#SBATCH --error=analysis/logs/baseline_features_expanded_%j.err

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/cme290/joaop/SLAC-atoms}"
cd "$PROJECT_DIR"
if [[ ! -f analysis/baseline_beamstop.py || ! -d data/trajectory_data_cap200_new ]]; then
    echo "ERROR: submit this job from the SLAC-atoms repo root." >&2
    echo "PROJECT_DIR resolved to: $PROJECT_DIR" >&2
    exit 1
fi

source ~/.bashrc
conda activate SLAC-p100

mkdir -p analysis/logs analysis/figures

PYTHONPATH="$PROJECT_DIR" python -u analysis/baseline_beamstop.py \
  --data-dir data/trajectory_data_cap200_new \
  --use-mol-features \
  --checkpoint models/beamstop/code/checkpoints/trial34_params/best_model.pt \
  --n-channels 128 \
  --R-cap 9.451833630105082 \
  --max-samples 50000 \
  --plot analysis/figures/beamstop_features_expanded_baseline_vs_model.png \
  --trajectory-heatmap analysis/figures/beamstop_features_expanded_trajectory_heatmaps.png
