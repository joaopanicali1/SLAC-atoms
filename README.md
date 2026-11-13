# SLAC-atoms

Tools for generating synthetic molecular trajectories, converting them into
pair-density and scattering data, training reconstruction models, and evaluating
low-q beamstop recovery and full pair-density recovery against the natural
scattering kernel (NSK) baseline (Natan, Phys. Rev. A 107, 023105 (2023)).

The repository is organized around five workflows:

1. Extract normal modes from OMol/data/train_4m structures.
2. Generate trajectory, pair-density (PD), and scattering (`S0`) datasets.
3. Train the UNet, learned primal-dual (LPD v1 and v2), and beamstop models.
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
    code/            LearnedPrimalDual (v1) and LearnedPrimalDualV2 + final .pt files
    training/        LPD v1/v2 training scripts and Slurm submit files
    dataset/         LPDDataset (v1) and LPDv2Dataset
  nsk/
    code/            NSK-l1 as published (Natan, PRA 2023): exact lasso path + L-curve
  unet/
    code/            PairDensityUNet and committed .pt files
    training/        UNet training script and Slurm submit file
    dataset/         MolScatteringDataset

analysis/
  beamstop/          final beamstop metrics, figures, and sensitivity analyses
  pd_recovery/       full-PD recovery: LPD v2 vs DBT/Tikhonov/NSK benchmark,
                     NSK paper Fig. 3 benchmark, older UNet/classical comparisons

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

### Learned Primal-Dual v2

LPD v2 recovers the normalised pair density `p(R) = pd_true(R) / sum_{a<b} Z_a Z_b`
from `S0(q)` measured over any q-range (random beamstop `q_min` during training).
Compared with v1 it uses the Debye equation as its forward operator, takes the
analysis window as an input so data consistency is only enforced at measured q,
starts from a Tikhonov solution, predicts a non-negative unit-mass output, and is
trained with a Wasserstein (W1) loss. The module docstring of
`models/lpd/code/learned_primal_dual_v2.py` explains each change.

It was trained on the original 662-molecule set `data/trajectory_data_cap200`
with `data/splits/split_lpd.npy`, the same set and split as LPD v1 (the beamstop
model uses the expanded `trajectory_data_cap200_new`).

```bash
sbatch models/lpd/training/submit_train_lpd_v2.slurm   # 8 GPUs (torchrun + DDP), --resume
```

Single device:

```bash
python models/lpd/training/train_lpd_v2.py \
  --data-dir data/trajectory_data_cap200 \
  --split data/splits/split_lpd.npy \
  --checkpoint-dir models/lpd/code/checkpoints/lpd_v2
```

The released checkpoint `models/lpd/code/lpd_v2_final.pt` is epoch 8 of that run
(best clean validation W1).

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

### Pair-density recovery (LPD v2)

The benchmark harness `analysis/pd_recovery/eval_pd_recovery.py` scores DBT,
Tikhonov, NSK-l1 as published (L-curve and tuned epsilon), LPD v1 and LPD v2 on
all 66 validation molecules of `split_lpd.npy` (20 frames each), in three
noise-free conditions:

```bash
mkdir -p analysis/logs
for i in 0 1 2; do sbatch --job-name=eval_pd_$i analysis/pd_recovery/submit_eval_pd.slurm $i; done
# 0 clean (q = 0-10 1/A), 1 beamstop q_min = 0.8 1/A, 2 Delta PD = PD(t) - PD(0)
```

Median results (R^2 after blurring prediction and truth by sigma = 0.025 / 0.05 / 0.1 A;
W1 in A):

| Condition | Tikhonov | NSK-l1 (paper) | LPD v2 |
|---|---|---|---|
| Clean | 0.52 / 0.73 / 0.91, W1 0.074 | 0.21 / 0.69 / 0.92, W1 0.099 | 0.74 / 0.90 / 0.98, W1 0.037 |
| Beamstop 0.8 | 0.22 / 0.34 / 0.47, W1 2.16 | 0.16 / 0.61 / 0.87, W1 0.270 | 0.73 / 0.90 / 0.98, W1 0.043 |

The NSK paper's own super-resolution test (its Fig. 3: two pair distances
approaching each other, with detector-pixel noise) is reproduced for NSK and
LPD v2 by:

```bash
sbatch analysis/pd_recovery/submit_nsk_fig3.slurm
```

Results and figures are in `analysis/pd_recovery/results/lpd_v2/`; see
`analysis/pd_recovery/README.md`.

Older full pair-density recovery comparisons for UNet and classical baselines
are also kept in `analysis/pd_recovery/`.

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

### Learned Primal-Dual v2

```python
import numpy as np
import torch
from models.lpd.code.learned_primal_dual_v2 import LearnedPrimalDualV2, process_scattering

ckpt = torch.load("models/lpd/code/lpd_v2_final.pt", map_location="cpu")
model = LearnedPrimalDualV2(ckpt["q"], ckpt["R_grid"], **ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

d = np.load("data/trajectory_data_cap200/mol_XXXXXX_pd.npz")
S0 = torch.tensor(d["S0"][0, :4])                       # 4 frames, (B, nq)
B = len(S0)
ss = torch.tensor(d["self_scatter"]).expand(B, -1)
fe = torch.tensor(d["fe"]).expand(B, -1)
q_min = torch.full((B,), 0.8)                           # beamstop edge [1/A]; 0 = full range
dS0, W = process_scattering(S0, ss, fe, torch.tensor(d["q"]), q_min)
with torch.no_grad():
    p_hat, _ = model(dS0, W)                            # normalised PD on R_grid, (B, nR)
```

### NSK-l1 (paper baseline)

```python
from models.nsk.code.nsk import NSK

# q (nq,), W (nq,) analysis window, x (n, nq) processed signal dS0, all numpy float64
nsk = NSK(q, W, 0.1)                           # dictionary spacing 0.1 A, as in the paper
stems, info = nsk.solve(x, positive=True)      # p = w/2 at nsk.R; epsilon at the L-curve corner
p = nsk.to_grid(stems, R_grid + dR / 2)        # onto the 0.025 A bins (dR = 0.025)
```

## Notes on Data Curation

The final expanded trajectory dataset contains generated molecules with at most
50 atoms. Molecules that failed normal-mode extraction, had no usable
vibrational modes, or produced unreadable trajectory files were skipped by the
pipeline. The normal-mode scripts optimize each structure before computing
frequencies; successful outputs contain the optimized coordinates, filtered
vibrational frequencies, and mode vectors used by downstream trajectory
generation.
