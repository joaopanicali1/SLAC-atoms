#!/bin/bash
#SBATCH --job-name=train_lpd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --output=training/logs/train_lpd_%j.out
#SBATCH --error=training/logs/train_lpd_%j.err

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source ~/.bashrc
conda activate SLAC

mkdir -p training/logs

PYTHONPATH="$PROJECT_DIR" \
python -u training/train_lpd.py
