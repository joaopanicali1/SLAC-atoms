#!/bin/bash
#SBATCH --job-name=bs_feat_new
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-pascal
#SBATCH --gres=gpu:1
#SBATCH --output=models/beamstop/training/logs/beamstop_features_expanded_%j.out
#SBATCH --error=models/beamstop/training/logs/beamstop_features_expanded_%j.err

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
if [[ ! -f models/beamstop/training/train_beamstop.py || ! -d data/trajectory_data_cap200_new ]]; then
    echo "ERROR: submit this job from the SLAC-atoms repo root." >&2
    echo "PROJECT_DIR resolved to: $PROJECT_DIR" >&2
    exit 1
fi

source ~/.bashrc
conda activate SLAC-p100

mkdir -p models/beamstop/training/logs models/beamstop/code/checkpoints/trial34_params

PYTHONPATH="$PROJECT_DIR" python -u models/beamstop/training/train_beamstop.py \
  --data-dir data/trajectory_data_cap200_new \
  --use-mol-features \
  --n-channels 128 \
  --lambda-boundary 0.45026251076764284 \
  --lambda-pd 1.8997338367460246e-05 \
  --R-cap 9.451833630105082 \
  --lr 0.0005 \
  --batch-size 128 \
  --num-workers 8 \
  --no-pin-memory \
  --n-epochs 30 \
  --n-warmup-epochs 2 \
  --max-runtime-min 215 \
  --stop-buffer-min 12 \
  --checkpoint-dir models/beamstop/code/checkpoints/trial34_params
