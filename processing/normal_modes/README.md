# Normal-Mode Extraction

This folder extracts optimized geometries and vibrational modes from
`data/train_4m/data0000.aselmdb`.

## Files

```text
classify.py        writes Slurm index arrays grouped by atom count
process_pyscf.py   small-molecule PySCF worker
process_orca.py    ORCA worker for medium/large molecules
modes_lib.py       shared PySCF/ORCA normal-mode utilities
submit_*.slurm     Slurm array launchers
indices/           current index arrays
```

## Typical Run

```bash
python processing/normal_modes/classify.py
sbatch processing/normal_modes/submit_small.slurm
sbatch processing/normal_modes/submit_medium.slurm
sbatch processing/normal_modes/submit_large.slurm
```

## Cluster Overrides

```bash
export SLAC_ATOMS_ASE_DB=/path/to/data0000.aselmdb
export SLAC_ATOMS_MOL_DATA_DIR=/path/to/data/mol_data_new
export OPENMPI_HOME=/path/to/openmpi
export ORCA_HOME=/path/to/orca_directory
export ORCA_EXE=/path/to/orca
```

Outputs are `mol_XXXXXX_physics.npz` files under `data/mol_data_new` by default.
