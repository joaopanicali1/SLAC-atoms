#!/bin/bash
#SBATCH --job-name=pd_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --output=/home/research/joaop/SLAC-atoms/processing/logs/pd_full_%j.out
#SBATCH --error=/home/research/joaop/SLAC-atoms/processing/logs/pd_full_%j.err

cd /home/research/joaop/SLAC-atoms
source ~/.bashrc
conda activate SLAC

mkdir -p data/trajectory_data_full

PYTHONPATH=/home/research/joaop/SLAC-atoms \
python processing/analyse_mol_v2.py \
    --out-dir data/trajectory_data_full
