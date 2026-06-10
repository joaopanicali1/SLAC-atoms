#!/bin/bash
#SBATCH --job-name=bs_ft34_new
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-pascal
#SBATCH --gres=gpu:1
#SBATCH --output=training/logs/finetune_34_expanded_%j.out
#SBATCH --error=training/logs/finetune_34_expanded_%j.err

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/cme290/joaop/SLAC-atoms}"
cd "$PROJECT_DIR"
if [[ ! -f models/beamstop/training/train_beamstop.py || ! -d data/trajectory_data_cap200_new ]]; then
    echo "ERROR: submit this job from the SLAC-atoms repo root." >&2
    echo "PROJECT_DIR resolved to: $PROJECT_DIR" >&2
    exit 1
fi

source ~/.bashrc
conda activate SLAC-p100

mkdir -p training/logs checkpoints/optuna_finetune_expanded/trial_0034

PYTHONPATH="$PROJECT_DIR" python -u models/beamstop/training/train_beamstop.py \
  --data-dir data/trajectory_data_cap200_new \
  --n-channels 128 \
  --lambda-boundary 0.45026251076764284 \
  --lambda-pd 1.8997338367460246e-05 \
  --R-cap 9.451833630105082 \
  --lr 0.00005 \
  --batch-size 128 \
  --num-workers 8 \
  --no-pin-memory \
  --n-epochs 30 \
  --n-warmup-epochs 1 \
  --max-runtime-min 215 \
  --stop-buffer-min 12 \
  --init-checkpoint checkpoints/optuna_finetune/trial_0034/best_model.pt \
  --checkpoint-dir checkpoints/optuna_finetune_expanded/trial_0034
