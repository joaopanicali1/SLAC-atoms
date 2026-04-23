#!/bin/bash
#SBATCH --job-name=orca_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=orca_test_r2scan%j.log

# 1. Activate your Conda environment
source ~/.bashrc
conda activate SLAC

export PATH="/home/cme290/joaop/SLAC-atoms/openmpi-4.1.8_install/bin:$PATH"
export LD_LIBRARY_PATH="/home/cme290/joaop/SLAC-atoms/openmpi-4.1.8_install/lib:$LD_LIBRARY_PATH"

# Path to ORCA
export PATH="/home/cme290/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH"
export LD_LIBRARY_PATH="/home/cme290/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$LD_LIBRARY_PATH"

# 2. Create the high-speed RAM Disk
export RAM_SCRATCH=/dev/shm/$USER_$SLURM_JOB_ID
mkdir -p $RAM_SCRATCH

echo "Running with RAM disk: $RAM_SCRATCH"

# 3. Execute your Python script
# Python will automatically detect $RAM_SCRATCH and use it
python -u orca_test.py

# 4. Clean up the RAM disk to be polite to the cluster
rm -rf $RAM_SCRATCH
echo "Job complete and RAM disk cleared."

