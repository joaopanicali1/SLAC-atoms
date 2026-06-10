#!/bin/bash
# Retry failed medium-tier molecules using Opt instead of TightOpt.
# Run collect_failed.py first to populate failed_medium_indices.npy,
# then submit this script.
#
# Purpose: test whether TightOpt convergence failure is the root cause
# of the ~32% failure rate. Successful retries save to the same mol_data/
# path, so they won't be reprocessed in future runs.
#SBATCH --job-name=modes_retry
#SBATCH --array=0-29%15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=processing/logs/retry_%A_%a.out
#SBATCH --error=processing/logs/retry_%A_%a.err

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

# Switch to Opt (standard convergence) instead of TightOpt
export ORCA_KEYWORDS="B3LYP D4 RIJCOSX def2-TZVPD def2/J TightSCF Opt FREQ"
export INDEX_FILE="processing/normal_modes/indices/failed_medium_indices.npy"

PYTHONPATH="$PROJECT_DIR" python -u processing/normal_modes/process_orca.py

rm -rf "$RAM_SCRATCH"
