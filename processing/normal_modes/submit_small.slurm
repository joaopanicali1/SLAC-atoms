#!/bin/bash
# Tier 1 — small molecules (≤6 atoms): PySCF, 16 cores each
# Two tasks fit on each 32-core node (%34 = 2 × 17 nodes)
# Update --array upper bound to (n_small - 1) after running classify.py
#SBATCH --job-name=modes_small
#SBATCH --array=0-8%34
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=processing/logs/small_%A_%a.out
#SBATCH --error=processing/logs/small_%A_%a.err

source ~/.bashrc
conda activate SLAC

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

export INDEX_FILE="processing/normal_modes/indices/small_indices.npy"

PYTHONPATH="$PROJECT_DIR" python -u processing/normal_modes/process_pyscf.py
