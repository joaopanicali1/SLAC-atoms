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

export PATH="/home/research/joaop/SLAC-atoms/openmpi-4.1.8_install/bin:$PATH"
export LD_LIBRARY_PATH="/home/research/joaop/SLAC-atoms/openmpi-4.1.8_install/lib:$LD_LIBRARY_PATH"
export PATH="/home/research/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH"
export LD_LIBRARY_PATH="/home/research/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$LD_LIBRARY_PATH"

# RAM disk for ORCA scratch (fast I/O, auto-cleaned at job end)
export RAM_SCRATCH=/dev/shm/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAM_SCRATCH"

export INDEX_FILE="processing/indices/medium_indices.npy"

python -u processing/process_orca.py

rm -rf "$RAM_SCRATCH"