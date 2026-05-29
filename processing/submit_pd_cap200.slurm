#!/bin/bash
#SBATCH --job-name=pd_cap200_new
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-pascal
#SBATCH --gres=gpu:1
#SBATCH --output=processing/logs/pd_cap200_%j.out
#SBATCH --error=processing/logs/pd_cap200_%j.err

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/cme290/joaop/SLAC-atoms}"
cd "$PROJECT_DIR"
if [[ ! -f processing/analyse_mol_v2.py || ! -d data/mol_data_new ]]; then
    echo "ERROR: submit this job from the SLAC-atoms repo root." >&2
    echo "PROJECT_DIR resolved to: $PROJECT_DIR" >&2
    exit 1
fi
source ~/.bashrc
conda activate SLAC

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p data/trajectory_data_cap200_new
echo "Prefilling expanded output directory with existing cap200 trajectory files..."
cp -al data/trajectory_data_cap200/. data/trajectory_data_cap200_new/ 2>/dev/null || \
    cp -an data/trajectory_data_cap200/. data/trajectory_data_cap200_new/
echo "Existing expanded outputs: $(find data/trajectory_data_cap200_new -maxdepth 1 -name 'mol_*_pd.npz' | wc -l)"

PYTHONPATH="$PROJECT_DIR" \
python processing/analyse_mol_v2.py \
    --data-dir data/mol_data_new \
    --out-dir data/trajectory_data_cap200_new \
    --max-frames 200
