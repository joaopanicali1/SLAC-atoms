#!/bin/bash
#SBATCH --job-name=beamstop_cond
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --output=training/logs/train_beamstop_cond_%j.out
#SBATCH --error=training/logs/train_beamstop_cond_%j.err

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source ~/.bashrc
conda activate SLAC

mkdir -p training/logs checkpoints/beamstop_conditioned

PYTHONPATH="$PROJECT_DIR" \
python -u models/beamstop/training/train_beamstop.py \
    --n-channels 128 \
    --use-conditioning \
    --lambda-masked 10.0 \
    --no-pin-memory \
    --num-workers 0 \
    --checkpoint-dir checkpoints/beamstop_conditioned
