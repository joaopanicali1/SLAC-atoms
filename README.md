# SLAC-atoms

Tools for generating synthetic molecular trajectories, converting them into
pair-density and scattering data, training reconstruction models, and evaluating
low-q beamstop recovery against the NSK baseline.

The repository is organized around five workflows:

1. Extract normal modes from OMol/data/train_4m structures.
2. Generate trajectory, pair-density (PD), and scattering (`S0`) datasets.
3. Train the UNet, learned primal-dual (LPD), and beamstop models.
4. Generate evaluation metrics and figures.
5. Load the best checkpoints for downstream inference.

Large generated artifacts are intentionally not committed. The expected local
data/checkpoint directories are documented below.

## Repository Map

```text
processing/
  normal_modes/      PySCF/ORCA normal-mode extraction from data/train_4m
  sampling/          phase-space sampling notebooks and sampled molecule ids
  form_factors/      x-ray atomic form factors used by analyse_mol_v2.py
  analyse_mol_v2.py  trajectory + PD + S0 generation from mol_data npz files
  regen_traj.py      resumable local trajectory regeneration helper
  submit_pd_*.slurm  Slurm wrappers for trajectory/scattering generation
  Archive/           historical scripts, benchmarks, and old experiments

data/
  train_4m*/          original train_4M ASE database (ignored by git)
  splits/            molecule-level train/validation split arrays
  mol_data_new/      generated normal-mode files
  trajectory_data*/  generated PD/S0 trajectory datasets (ignored by git)

models/
  beamstop/
    code/            BeamstopModel and final .pt/.json
    training/        beamstop training script and Slurm submit file
    dataset/         BeamstopDataset
  lpd/
    code/            LearnedPrimalDual implementation
    training/        LPD training script and Slurm submit file
    dataset/         LPDDataset
  unet/
    code/            PairDensityUNet and committed .pt files
    training/        UNet training script and Slurm submit file
    dataset/         MolScatteringDataset

analysis/
  beamstop/          final beamstop metrics, figures, and sensitivity analyses
  pd_recovery/       UNet/classical full-PD recovery metrics and figures

archive/
  engine_validation/ historical B3LYP validation molecules/results
  modal/             historical Modal training/evaluation scripts
```

## Environment Notes

The project was run with two Conda-style environments:

- `SLAC`: preprocessing and CPU/scientific Python jobs.
- `SLAC-p100`: GPU training/evaluation on Pascal P100 nodes with a PyTorch build
  compatible with compute capability 6.0.

Set `PYTHONPATH` to the repository root when running scripts directly:

```bash
cd /path/to/SLAC-atoms
export PYTHONPATH="$PWD"
```

The Slurm files assume they are submitted from the repository root and use
`$SLURM_SUBMIT_DIR`, so they should not need path edits when cloned elsewhere.

## 1. Extract Normal Modes

Normal-mode extraction starts from the original ASE database shards:

```text
data/train_4m/*.aselmdb
```

Create molecule index arrays grouped by atom count:

```bash
python processing/normal_modes/classify.py
```

This writes:

```text
processing/normal_modes/indices/{small,medium,large,xlarge}_indices.npy
```

By default, `classify.py` reads the sampled molecule ids and feature table from
`processing/sampling/`. That folder contains the notebooks, figures, and `.npy`
files used to build the phase-space sample from `data/train_4m`.

Run the extraction workers through Slurm:

```bash
sbatch processing/normal_modes/submit_small.slurm    # PySCF, small molecules
sbatch processing/normal_modes/submit_medium.slurm   # ORCA, medium molecules
sbatch processing/normal_modes/submit_large.slurm    # ORCA, large molecules
```

The active ORCA/PySCF implementation is in:

```text
processing/normal_modes/process_orca.py
processing/normal_modes/process_pyscf.py
processing/normal_modes/modes_lib.py
```

Outputs are saved as:

```text
data/mol_data_new/mol_XXXXXX_physics.npz
```

Useful overrides:

```bash
export SLAC_ATOMS_ASE_DB=/path/to/data0000.aselmdb
export SLAC_ATOMS_MOL_DATA_DIR=/path/to/output/mol_data_new
export ORCA_HOME=/path/to/orca_directory
export ORCA_EXE=/path/to/orca
export OPENMPI_HOME=/path/to/openmpi
```

## 2. Generate Trajectory, PD, and S0 Data

Use `processing/analyse_mol_v2.py` on the generated normal-mode files:

```bash
python processing/analyse_mol_v2.py \
  --data-dir data/mol_data_new \
  --out-dir data/trajectory_data_cap200_new \
  --max-frames 200
```

On the cluster, submit:

```bash
sbatch processing/submit_pd_cap200.slurm
```

`analyse_mol_v2.py` uses `processing/form_factors/atomic_form_factors.py` to
compute Waasmaier-Kirfel x-ray atomic form factors. Those form factors weight
the synthetic scattering signal `S0(q,t)` before the finite-q pair-density
quantities are computed.

For local resumable regeneration without Slurm, use:

```bash
python processing/regen_traj.py \
  --data-dir data/mol_data_new \
  --out-dir data/trajectory_data_v2 \
  --max-frames 500 \
  --workers 8
```

`regen_traj.py` calls the same molecule processor as `analyse_mol_v2.py` and
skips outputs that already exist, so it is safe to restart after interruption.

Each output file is:

```text
data/trajectory_data_cap200_new/mol_XXXXXX_pd.npz
```

Important arrays include:

- `q`, `R_grid`, `time_fs`
- `S0`: resolution-limited scattering curves
- `dS0`: finite-q normalized scattering signal used by LPD
- `pd_true`: direct pair-density histograms from atomic coordinates
- `pd_limited`: resolution-limited PD from the finite-q signal
- `V_vib`, `omegas_ASE`: normal-mode metadata

The beamstop dataset computes its supervised target from `S0` as:

```text
Delta S0_norm(q,t) = (S0(q,t) - S0(q, frame 0)) / fe(q)
```

Frame 0 is excluded because the differential signal is exactly zero there.

## 3. Train Models

### UNet

```bash
python models/unet/training/train_unet.py \
  --data-dir data/trajectory_data_cap200_new \
  --checkpoint-dir models/unet/code/checkpoints
```

or:

```bash
sbatch models/unet/training/submit_train_unet.slurm
```

The UNet maps `pd_limited -> pd_true` and uses normal-mode arrays as auxiliary
inputs.

### Learned Primal-Dual

```bash
python models/lpd/training/train_lpd.py \
  --data-dir data/trajectory_data_cap200_new \
  --checkpoint-dir models/lpd/code/checkpoints \
  --no-pin-memory
```

or:

```bash
sbatch models/lpd/training/submit_train_lpd.slurm
```

The LPD model maps `dS0(q) -> pd_true(R)` using a fixed DBT operator plus
learned unrolled updates.

### Beamstop

Final beamstop training used the expanded cap200 dataset, molecular scalar
conditioning, and the hyperparameters saved in `models/beamstop/code/beamstop_final.json`.

```bash
sbatch models/beamstop/training/submit_train_beamstop_features_expanded.slurm
```

Direct Python equivalent:

```bash
python models/beamstop/training/train_beamstop.py \
  --data-dir data/trajectory_data_cap200_new \
  --use-mol-features \
  --n-channels 128 \
  --R-cap 9.451833630105082 \
  --lambda-boundary 0.45026251076764284 \
  --lambda-pd 1.8997338367460246e-05 \
  --lr 0.0005 \
  --batch-size 128 \
  --num-workers 8 \
  --no-pin-memory \
  --checkpoint-dir models/beamstop/code/checkpoints/trial34_params
```

## 4. Evaluate Models and Generate Figures

Final three-way beamstop evaluation compares:

- beamstop conditioned (`models/beamstop/code/beamstop_final.pt`)
- beamstop unconditioned (`checkpoints/optuna_finetune_expanded/trial_0034/best_model.pt`)
- NSK baseline (`analysis/beamstop/baseline_beamstop.py`)

Run:

```bash
sbatch analysis/beamstop/submit_eval_beamstop_features_benchmark.slurm
```

Outputs are written to:

```text
analysis/beamstop/final_results/beamstop_conditioned_<job_id>/
```

The main outputs are:

- `summary_metrics.csv`
- `per_sample_metrics.csv`
- `per_molecule_metrics.csv`
- `q_resolved_rmse.csv`
- representative, worst, and conditioned-beats-NSK plots
- MSE boxplot and scatter plots

Sensitivity analyses:

```bash
sbatch analysis/beamstop/submit_eval_beamstop_qmin_sensitivity.slurm
sbatch analysis/beamstop/submit_eval_beamstop_snr_sensitivity.slurm
```

These produce CSV files and MSE curves under:

```text
analysis/beamstop/final_results/qmin_sensitivity_<job_id>/
analysis/beamstop/final_results/snr_sensitivity_<job_id>/
```

Full pair-density recovery comparisons for UNet and classical baselines are
kept in:

```text
analysis/pd_recovery/
```

## 5. Load Best Checkpoints

### Beamstop Final Model

```python
import json
import torch

from models.beamstop.dataset.dataset_beamstop import BeamstopDataset
from models.beamstop.code.beamstop import BeamstopModel, build_sinc_operator
from models.beamstop.training.train_beamstop import compute_dataset_R_max, build_truncated_R_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("models/beamstop/code/beamstop_final.json") as f:
    meta = json.load(f)

dataset = BeamstopDataset(
    meta["dataset"],
    q_min_lo=0.5,
    q_min_hi=1.0,
    max_atoms=75,
    max_modes=219,
    use_mol_features=True,
)

R_max_data = compute_dataset_R_max(dataset.loaded_files, margin=1.2)
dR = float(dataset.R_grid[1] - dataset.R_grid[0])
R_grid = build_truncated_R_grid(min(R_max_data, meta["R_cap"]), dR=dR)
B_sinc = build_sinc_operator(dataset.q.numpy(), R_grid, verbose=False).to(device)

model = BeamstopModel(
    B_sinc,
    n_channels=meta["n_channels"],
    use_feature_conditioning=True,
    mol_feature_dim=meta["mol_feature_dim"],
).to(device)

ckpt = torch.load(meta["checkpoint"], map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

### UNet

```python
import torch
from models.unet.code.unet import PairDensityUNet

model = PairDensityUNet(
    n_R=1257,
    features=[32, 64, 128, 256],
    d_model=256,
    n_heads=8,
    max_atoms=50,
    max_modes=144,
)
ckpt = torch.load("models/unet/code/unet_best_model_epoch208.pt", map_location="cpu")
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()
```

### Learned Primal-Dual

```python
import torch
from models.lpd.dataset.dataset_lpd import LPDDataset
from models.lpd.code.learned_primal_dual import LearnedPrimalDual, build_dbt_operator

dataset = LPDDataset("data/trajectory_data_cap200_new")
A_dbt, _ = build_dbt_operator(dataset.q.numpy(), len(dataset.R_grid) + 1)
model = LearnedPrimalDual(A_dbt)

ckpt = torch.load("models/lpd/code/lpd_final.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

## Notes on Data Curation

The final expanded trajectory dataset contains generated molecules with at most
50 atoms. Molecules that failed normal-mode extraction, had no usable
vibrational modes, or produced unreadable trajectory files were skipped by the
pipeline. The normal-mode scripts optimize each structure before computing
frequencies; successful outputs contain the optimized coordinates, filtered
vibrational frequencies, and mode vectors used by downstream trajectory
generation.
