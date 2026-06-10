# Processing

This folder contains the data-generation pipeline.

```text
normal_modes/    active PySCF/ORCA normal-mode extraction code
sampling/        molecule phase-space sampling notebooks and selected ids
form_factors/    x-ray atomic form factor tables used during scattering generation
analyse_mol_v2.py trajectory, PD, and S0 generation
regen_traj.py    resumable local wrapper around analyse_mol_v2.process_molecule
Archive/         historical processing experiments
```

## Normal Modes

Active normal-mode extraction code lives in `processing/normal_modes/`.

```bash
python processing/normal_modes/classify.py
sbatch processing/normal_modes/submit_small.slurm
sbatch processing/normal_modes/submit_medium.slurm
sbatch processing/normal_modes/submit_large.slurm
```

`classify.py` reads the default sample and molecular feature table from
`processing/sampling/`. Override those paths with `SLAC_ATOMS_SAMPLE_PATH` and
`SLAC_ATOMS_FEATURES_PATH` when using a different molecule sample.

Outputs:

```text
data/mol_data_new/mol_XXXXXX_physics.npz
```

The ORCA/PySCF implementation is shared through
`processing/normal_modes/modes_lib.py`.

## Sampling

`processing/sampling/` contains the notebooks, sample arrays, and diagnostic
figures used to select molecule ids from `data/train_4m`. The selected `.npy`
files drive `processing/normal_modes/classify.py`; they are not used directly by
training or evaluation.

## Trajectory and Scattering Generation

`analyse_mol_v2.py` converts normal-mode files into trajectory, PD, and S0
datasets:

```bash
python processing/analyse_mol_v2.py \
  --data-dir data/mol_data_new \
  --out-dir data/trajectory_data_cap200_new \
  --max-frames 200
```

During scattering generation, `analyse_mol_v2.py` imports
`processing/form_factors/atomic_form_factors.py` to evaluate x-ray atomic form
factors on the q grid. `processing/form_factors/form_factors.py` is kept as the
raw/helper source used to build that table.

Cluster wrapper:

```bash
sbatch processing/submit_pd_cap200.slurm
```

For a local resumable run without Slurm:

```bash
python processing/regen_traj.py \
  --data-dir data/mol_data_new \
  --out-dir data/trajectory_data_v2 \
  --max-frames 500 \
  --workers 8
```

`regen_traj.py` calls the same per-molecule processor and skips files whose
`mol_*_pd.npz` output already exists.

Historical scripts and benchmarking experiments are kept in `processing/Archive/`.
