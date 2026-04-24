#!/bin/bash
# Tier 3 — large molecules (13-50 atoms): ORCA, 32 cores, 1 node per task
# Update --array upper bound to (n_large - 1) after running classify.py
#SBATCH --job-name=modes_large
#SBATCH --array=0-423%17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=processing/logs/large_%A_%a.out
#SBATCH --error=processing/logs/large_%A_%a.err

source ~/.bashrc
conda activate SLAC

export PATH="/home/cme290/joaop/SLAC-atoms/openmpi-4.1.8_install/bin:$PATH"
export LD_LIBRARY_PATH="/home/cme290/joaop/SLAC-atoms/openmpi-4.1.8_install/lib:$LD_LIBRARY_PATH"
export PATH="/home/cme290/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH"
export LD_LIBRARY_PATH="/home/cme290/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$LD_LIBRARY_PATH"

export RAM_SCRATCH=/dev/shm/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAM_SCRATCH"

export INDEX_FILE="processing/indices/large_indices.npy"

python -u processing/process_orca.py

rm -rf "$RAM_SCRATCH"