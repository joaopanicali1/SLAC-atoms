#!/bin/bash
# Tier 2 — medium molecules (7-12 atoms): ORCA, 32 cores, 1 node per task
# %17 = one task per node across 17 available nodes
# Update --array upper bound to (n_medium - 1) after running classify.py
#SBATCH --job-name=modes_medium
#SBATCH --array=0-136%17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=processing/logs/medium_%A_%a.out
#SBATCH --error=processing/logs/medium_%A_%a.err

source ~/.bashrc
conda activate SLAC

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

OPENMPI_HOME="${OPENMPI_HOME:-$PROJECT_DIR/openmpi-4.1.8_install}"
ORCA_HOME="${ORCA_HOME:-$PROJECT_DIR/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg}"
export PATH="$OPENMPI_HOME/bin:$ORCA_HOME:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib:$ORCA_HOME:${LD_LIBRARY_PATH:-}"
export ORCA_EXE="${ORCA_EXE:-$ORCA_HOME/orca}"

# RAM disk for ORCA scratch (fast I/O, auto-cleaned at job end)
export RAM_SCRATCH=/dev/shm/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAM_SCRATCH"

export INDEX_FILE="processing/normal_modes/indices/medium_indices.npy"

PYTHONPATH="$PROJECT_DIR" python -u processing/normal_modes/process_orca.py

rm -rf "$RAM_SCRATCH"
