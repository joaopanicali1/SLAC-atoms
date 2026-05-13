#!/bin/bash
# Submit one independent SLURM job per molecule.
# Molecules whose .npz already exists are skipped automatically.
# Run from the project root: bash processing/Benchmarking/submit_all_configs.sh

cd "$(dirname "$0")/../.."

sbatch --export=MOL=H2O_3atoms     --job-name=bench_H2O   processing/Benchmarking/submit_benchmark_configs.slurm
sbatch --export=MOL=CH3NO2_7atoms  --job-name=bench_NM    processing/Benchmarking/submit_benchmark_configs.slurm
sbatch --export=MOL=C3HAuO2_7atoms  --job-name=bench_Au   processing/Benchmarking/submit_benchmark_configs.slurm

echo "Jobs submitted. Molecules with existing results will exit immediately."
echo "Track with: squeue -u \$USER"
echo "When complete, run:"
echo "  python processing/Benchmarking/benchmark_configs_analyze.py"
