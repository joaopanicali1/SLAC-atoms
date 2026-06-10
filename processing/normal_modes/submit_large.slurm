#!/bin/bash
# Tier 3 — large molecules (13-50 atoms): ORCA, 2 nodes / 64 cores per task
#
# Benchmark results:
#   25-atom organic:    59 min on 1-node → fine on 1-node
#   40-atom Pt complex: timed out on 1-node, 164 min on 2-node
# Using 2 nodes for the whole tier protects against hard molecules timing out.
# OOM tests show 4+ nodes causes failures for this molecule size range.
#
# Update --array upper bound to (n_large - 1) after running classify.py
#SBATCH --job-name=modes_large
#SBATCH --array=0-423%7
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=processing/logs/large_%A_%a.out
#SBATCH --error=processing/logs/large_%A_%a.err

source ~/.bashrc
conda activate SLAC

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

OPENMPI_HOME="${OPENMPI_HOME:-$PROJECT_DIR/openmpi-4.1.8_install}"
ORCA_HOME="${ORCA_HOME:-$PROJECT_DIR/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg}"
export PATH="$OPENMPI_HOME/bin:$ORCA_HOME:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib:$ORCA_HOME:${LD_LIBRARY_PATH:-}"
export ORCA_EXE="${ORCA_EXE:-$ORCA_HOME/orca}"

export RAM_SCRATCH=/dev/shm/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAM_SCRATCH"

export INDEX_FILE="processing/normal_modes/indices/large_indices.npy"

PYTHONPATH="$PROJECT_DIR" python -u processing/normal_modes/process_orca.py

rm -rf "$RAM_SCRATCH"
