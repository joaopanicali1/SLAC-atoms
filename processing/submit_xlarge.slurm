#!/bin/bash
# Tier 4 — extra-large molecules (>50 atoms): ORCA, 2 nodes / 64 cores per task
#
# Benchmark results:
#   55-atom:  74 min on 2-node
#   72-atom: 163 min on 2-node
#   100-atom: timed out on 2-node (complex molecule near 4-hr limit)
#   4-node (128 cores): OUT OF MEMORY — too many MPI processes per basis function
#   8-node (256 cores): ORCA internal error (negative matrix dims)
# 2 nodes is the maximum that works reliably for this molecule size range.
# Molecules near 100+ atoms may still time out; resubmit those individually.
#
# Update --array upper bound to (n_xlarge - 1) after running classify.py
#SBATCH --job-name=modes_xlarge
#SBATCH --array=0-429%7
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --exclude=hpcc-cpu-11-16,hpcc-cpu-11-18
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