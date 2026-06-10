#!/bin/bash
#SBATCH --job-name=beamstop_128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-volta
#SBATCH --gres=gpu:1
#SBATCH --output=training/logs/train_beamstop_128_%j.out
#SBATCH --error=training/logs/train_beamstop_128_%j.err

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source ~/.bashrc
conda activate SLAC

mkdir -p training/logs checkpoints/beamstop_128

PYTHONPATH="$PROJECT_DIR" \
python -u models/beamstop/training/train_beamstop.py \
    --n-channels 128 \
    --batch-size 128 \
    --lr 2e-3 \
    --lambda-smooth 0 \
    --lambda-boundary 1.0 \
    --lambda-pd 1e-3 \
    --R-cap 10.0 \
    --checkpoint-dir checkpoints/beamstop_128
