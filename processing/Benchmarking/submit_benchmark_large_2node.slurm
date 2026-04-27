#!/bin/bash
# Benchmark: large tier (13-50 atoms), 2 nodes / 64 cores
# Array tasks 0-2 map to the same molecules as the 1-node run for direct comparison
#SBATCH --job-name=bench_large_2n
#SBATCH --array=0-2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --exclude=hpcc-cpu-11-16,hpcc-cpu-11-18
#SBATCH --output=processing/logs/bench_large_2n_%A_%a.out
#SBATCH --error=processing/logs/bench_large_2n_%A_%a.err

source ~/.bashrc
conda activate SLAC

export PATH="/home/research/joaop/SLAC-atoms/openmpi-4.1.8_install/bin:$PATH"
export LD_LIBRARY_PATH="/home/research/joaop/SLAC-atoms/openmpi-4.1.8_install/lib:$LD_LIBRARY_PATH"
export PATH="/home/research/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH"
export LD_LIBRARY_PATH="/home/research/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$LD_LIBRARY_PATH"

# ORCA handles multi-node internally via SLURM_JOB_NODELIST — workers communicate
# purely via MPI and only the master process touches the scratch directory,
# so /dev/shm (node-local RAM disk) works fine.
export RAM_SCRATCH=/dev/shm/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAM_SCRATCH"

python -u processing/benchmark.py --tier large --mol-idx "$SLURM_ARRAY_TASK_ID"

rm -rf "$RAM_SCRATCH"