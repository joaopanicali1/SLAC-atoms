#!/bin/bash
#SBATCH --job-name=pyscf_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=pyscf_test_%j.log

# 1. Activate your Conda environment
source ~/.bashrc
conda activate SLAC

python -u pyscf_test.py


