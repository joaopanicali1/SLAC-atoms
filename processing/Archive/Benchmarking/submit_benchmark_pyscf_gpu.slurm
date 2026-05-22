#!/bin/bash
# Benchmark: PySCF with GPU acceleration (gpu4pyscf)
# Runs all 6 molecules sequentially in one job (cluster MaxArraySize=2):
#   0: C2H8N4P       (15 atoms)
#   1: C6H17NO       (25 atoms)
#   2: PtC11H18N3O6P (40 atoms)
#   3: Br2C15H32IN2O3(55 atoms)
#   4: C18H42N6O6    (72 atoms)
#   5: C25ClH49N7O17S(100 atoms)
#SBATCH --job-name=bench_pyscf_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-turing
#SBATCH --time=24:00:00
#SBATCH --output=/home/research/joaop/SLAC-atoms/processing/logs/bench_pyscf_gpu_%j.out
#SBATCH --error=/home/research/joaop/SLAC-atoms/processing/logs/bench_pyscf_gpu_%j.err

cd /home/research/joaop/SLAC-atoms

source ~/.bashrc
conda activate SLAC

module load cuda/12.2

export LD_LIBRARY_PATH="/home/research/joaop/.conda/envs/SLAC/lib/python3.12/site-packages/cutensor/lib:$LD_LIBRARY_PATH"

for mol_idx in 0 1 2 3 4 5; do
    echo "=== Starting mol-idx ${mol_idx} ==="
    python -u processing/Benchmarking/benchmark_pyscf_gpu.py --mol-idx "$mol_idx"
    echo "=== Done mol-idx ${mol_idx} ==="
done
