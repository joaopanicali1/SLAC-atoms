#!/bin/bash
# Tier 4 — extra-large molecules (>50 atoms): ORCA, 32 cores, 1 node per task
#
# WARNING: With 50-150 atoms, individual calculations may take several hours.
# The 4-hour wall-time limit on the cpu partition may be insufficient for the
# largest molecules (~100-150 atoms). Options if jobs time out:
#   a) Request a longer time limit from your admin (--time=08:00:00 or more)
#   b) Split xlarge further and allocate 2 nodes × 32 cores for the very largest
#
# Update --array upper bound to (n_xlarge - 1) after running classify.py
#SBATCH --job-name=modes_xlarge
#SBATCH --array=0-429%17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=processing/logs/xlarge_%A_%a.out
#SBATCH --error=processing/logs/xlarge_%A_%a.err

source ~/.bashrc
conda activate SLAC

export PATH="/home/cme290/joaop/SLAC-atoms/openmpi-4.1.8_install/bin:$PATH"
export LD_LIBRARY_PATH="/home/cme290/joaop/SLAC-atoms/openmpi-4.1.8_install/lib:$LD_LIBRARY_PATH"
export PATH="/home/cme290/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH"
export LD_LIBRARY_PATH="/home/cme290/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$LD_LIBRARY_PATH"

export RAM_SCRATCH=/dev/shm/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAM_SCRATCH"

export INDEX_FILE="processing/indices/xlarge_indices.npy"

python -u processing/process_orca.py

rm -rf "$RAM_SCRATCH"