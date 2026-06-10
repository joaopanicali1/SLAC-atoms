# Sampling

This folder contains the phase-space sampling materials used before normal-mode
extraction.

The notebooks and `sample.py` inspect molecules from `data/train_4m`, build a
feature table such as `phase_space_shard0.npy`, and write sampled molecule-id
arrays under `samples/`. The normal-mode classifier reads those arrays to create
Slurm index files:

```bash
python processing/normal_modes/classify.py
```

Useful files:

- `phase_space_shard0.npy`: per-molecule feature table used by `classify.py`.
- `samples/*.npy`: sampled molecule ids for normal-mode extraction.
- `figures/`: diagnostics showing the phase-space coverage of the sample.

Training and evaluation scripts do not consume this folder directly; it is part
of the data-curation stage that selects which molecules enter
`data/mol_data_new/`.
