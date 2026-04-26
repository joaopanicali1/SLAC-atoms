#!/bin/bash
# Benchmark: xlarge tier (>50 atoms), 8 nodes / 256 cores
# Only running mol-idx=2 (100 atoms) — the molecule that timed out on 4 nodes.
#SBATCH --job-name=bench_xl_8n
#SBATCH --array=2
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --exclude=hpcc-cpu-11-16,hpcc-cpu-11-18
#SBATCH --output=processing/logs/bench_xl_8n_%A_%a.out
#SBATCH --error=processing/logs/bench_xl_8n_%A_%a.err

source ~/.bashrc
conda activate SLAC

export PATH="/home/cme290/joaop/SLAC-atoms/openmpi-4.1.8_install/bin:$PATH"
export LD_LIBRARY_PATH="/home/cme290/joaop/SLAC-atoms/openmpi-4.1.8_install/lib:$LD_LIBRARY_PATH"
export PATH="/home/cme290/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH"
export LD_LIBRARY_PATH="/home/cme290/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$LD_LIBRARY_PATH"

export RAM_SCRATCH=/dev/shm/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAM_SCRATCH"

python -u processing/benchmark.py --tier xlarge --mol-idx "$SLURM_ARRAY_TASK_ID"

rm -rf "$RAM_SCRATCH"