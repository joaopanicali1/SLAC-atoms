#!/bin/bash
# Run all ORCA configs for ONE molecule (set via --export MOL=... at submission).
# Three of these jobs run simultaneously — one per molecule.
# ORCA runs single-core; gives full node RAM to ORCA_MAXCORE.
#SBATCH --job-name=bench_cfg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --output=/home/research/joaop/SLAC-atoms/processing/logs/bench_configs_%x_%j.out
#SBATCH --error=/home/research/joaop/SLAC-atoms/processing/logs/bench_configs_%x_%j.err

cd /home/research/joaop/SLAC-atoms

source ~/.bashrc
conda activate SLAC

export PATH="/home/research/joaop/SLAC-atoms/openmpi-4.1.8_install/bin:$PATH"
export LD_LIBRARY_PATH="/home/research/joaop/SLAC-atoms/openmpi-4.1.8_install/lib:$LD_LIBRARY_PATH"
export PATH="/home/research/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH"
export LD_LIBRARY_PATH="/home/research/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$LD_LIBRARY_PATH"

python -u processing/Benchmarking/benchmark_configs.py --mol "$MOL"
