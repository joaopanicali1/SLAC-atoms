#!/bin/bash
#SBATCH --job-name=pd_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=processing/logs/pd_full_%j.out
#SBATCH --error=processing/logs/pd_full_%j.err

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source ~/.bashrc
conda activate SLAC

mkdir -p data/trajectory_data_full

PYTHONPATH="$PROJECT_DIR" \
python processing/analyse_mol_v2.py \
    --out-dir data/trajectory_data_full
