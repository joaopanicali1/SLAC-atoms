#!/bin/bash
#SBATCH --job-name=beamstop_128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-volta
#SBATCH --gres=gpu:1
#SBATCH --output=/home/research/joaop/SLAC-atoms/training/logs/train_beamstop_128_%j.out
#SBATCH --error=/home/research/joaop/SLAC-atoms/training/logs/train_beamstop_128_%j.err

cd /home/research/joaop/SLAC-atoms
source ~/.bashrc
conda activate SLAC

mkdir -p training/logs checkpoints/beamstop_128

PYTHONPATH=/home/research/joaop/SLAC-atoms \
python -u training/train_beamstop.py \
    --n-channels 128 \
    --batch-size 128 \
    --lr 2e-3 \
    --lambda-smooth 0 \
    --lambda-boundary 1.0 \
    --lambda-pd 1e-3 \
    --R-cap 10.0 \
    --checkpoint-dir checkpoints/beamstop_128
