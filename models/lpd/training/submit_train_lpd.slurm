#!/bin/bash
#SBATCH --job-name=train_lpd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --output=models/lpd/training/logs/train_lpd_%j.out
#SBATCH --error=models/lpd/training/logs/train_lpd_%j.err

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
source ~/.bashrc
conda activate SLAC

mkdir -p models/lpd/training/logs models/lpd/code/checkpoints

PYTHONPATH="$PROJECT_DIR" \
python -u models/lpd/training/train_lpd.py \
    --data-dir data/trajectory_data_cap200_new \
    --checkpoint-dir models/lpd/code/checkpoints \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
    --no-pin-memory
