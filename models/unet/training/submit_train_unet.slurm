#!/bin/bash
# Train the UNet pair-density model on generated trajectory/PD data.
# Submit from the repository root:
#   sbatch models/unet/training/submit_train_unet.slurm
# Adjust partition/env names for the target cluster if needed.
#SBATCH --job-name=train_unet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-pascal
#SBATCH --gres=gpu:1
#SBATCH --output=models/unet/training/logs/train_unet_%j.out
#SBATCH --error=models/unet/training/logs/train_unet_%j.err

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
if [[ ! -f models/unet/training/train_unet.py || ! -d data/trajectory_data_cap200_new ]]; then
    echo "ERROR: submit this job from the SLAC-atoms repo root." >&2
    echo "PROJECT_DIR resolved to: $PROJECT_DIR" >&2
    exit 1
fi

source ~/.bashrc
conda activate SLAC-p100

mkdir -p models/unet/training/logs models/unet/code/checkpoints

PYTHONPATH="$PROJECT_DIR" python -u models/unet/training/train_unet.py \
  --data-dir data/trajectory_data_cap200_new \
  --checkpoint-dir models/unet/code/checkpoints \
  --batch-size 32 \
  --num-workers 4 \
  --no-pin-memory
