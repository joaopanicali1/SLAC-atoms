# Data

Generated data is not committed to git. Expected local directories:

```text
train_4m/                  original train_4M ASE database
mol_data_new/              expanded normal-mode set
trajectory_data_cap200/    original trajectory/PD/S0 dataset (662 molecules; LPD v1/v2)
trajectory_data_cap200_new/ expanded trajectory/PD/S0 dataset (887 molecules; beamstop)
```

Model-specific PyTorch datasets live with their models:

```text
models/unet/dataset/       UNet: pd_limited -> pd_true
models/lpd/dataset/        LPD: dS0(q) -> pd_true(R) (v1), S0(q) -> normalised p(R) (v2)
models/beamstop/dataset/   beamstop: masked Delta S0_norm(q) -> full Delta S0_norm(q)
```

The trajectory files are produced by `processing/analyse_mol_v2.py` and named
`mol_XXXXXX_pd.npz`.

`data/splits/` stores molecule-level shuffled index arrays. For each split file,
the first 10% of molecule indices are validation and the remaining 90% are
training unless a script says otherwise.
