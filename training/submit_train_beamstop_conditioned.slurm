#!/bin/bash
#SBATCH --job-name=beamstop_cond
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --output=/home/research/joaop/SLAC-atoms/training/logs/train_beamstop_cond_%j.out
#SBATCH --error=/home/research/joaop/SLAC-atoms/training/logs/train_beamstop_cond_%j.err

cd /home/research/joaop/SLAC-atoms
source ~/.bashrc
conda activate SLAC

mkdir -p training/logs checkpoints/beamstop_conditioned

PYTHONPATH=/home/research/joaop/SLAC-atoms \
python -u training/train_beamstop.py \
    --n-channels 128 \
    --use-conditioning \
    --lambda-masked 10.0 \
    --no-pin-memory \
    --checkpoint-dir checkpoints/beamstop_conditioned
