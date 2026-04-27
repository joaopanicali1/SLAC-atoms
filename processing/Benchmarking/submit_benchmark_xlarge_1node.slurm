#!/bin/bash
# Benchmark: xlarge tier (>50 atoms), 1 node / 32 cores
# Each array task is one molecule (~55, ~75, ~100 atoms); full 4 hours available per molecule
#SBATCH --job-name=bench_xl_1n
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=processing/logs/bench_xl_1n_%A_%a.out
#SBATCH --error=processing/logs/bench_xl_1n_%A_%a.err

source ~/.bashrc
conda activate SLAC

export PATH="/home/research/joaop/SLAC-atoms/openmpi-4.1.8_install/bin:$PATH"
export LD_LIBRARY_PATH="/home/research/joaop/SLAC-atoms/openmpi-4.1.8_install/lib:$LD_LIBRARY_PATH"
export PATH="/home/research/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH"
export LD_LIBRARY_PATH="/home/research/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$LD_LIBRARY_PATH"

export RAM_SCRATCH=/dev/shm/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAM_SCRATCH"

python -u processing/benchmark.py --tier xlarge --mol-idx "$SLURM_ARRAY_TASK_ID"

rm -rf "$RAM_SCRATCH"