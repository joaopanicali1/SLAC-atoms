# Learned Primal-Dual

Learns an unrolled reconstruction from q-space scattering to pair density.
Two versions share this folder, the original 662-molecule dataset
`data/trajectory_data_cap200`, and the split `data/splits/split_lpd.npy`.

```text
code/       LearnedPrimalDual (v1), LearnedPrimalDualV2, final checkpoints
training/   train_lpd.py / train_lpd_v2.py and their Slurm submit files
dataset/    LPDDataset (v1), LPDv2Dataset + split_files (v2)
```

## v1

`learned_primal_dual.py`: DBT-based operator, maps `dS0(q) -> pd_true(R)`
(absolute PD), full q-range only.

```text
models/lpd/code/lpd_final.pt
```

```bash
sbatch models/lpd/training/submit_train_lpd.slurm
```

## v2

`learned_primal_dual_v2.py`: maps `S0(q)` over any measured q-range to the
normalised pair density `p(R) = pd_true(R) / sum_{a<b} Z_a Z_b`. What v1 got
wrong and what v2 changes:

- **Forward operator.** v1's operator was not the Debye equation (it left
  50-80 % of `dS0` unexplained even for the true PD); v2 uses
  `dS0(q) = W(q) sum_m 2 j0(q Rc_m) p_m`.
- **Beamstop.** v2 takes the analysis window `W(q)` as an input, so data
  consistency is enforced only at measured q and one model handles any `q_min`
  (v1 failed at `q_min = 0.8`).
- **Warm start.** Tikhonov solution for each sample's own window (v1's was
  about 1e-6 of the target).
- **Scale.** `dS0` is divided by `fe ~ sum ZZ`, so the absolute scale is not
  recoverable; v2 predicts a non-negative, unit-mass `p`.
- **Loss.** 1-D Wasserstein (W1) instead of MSE, plus a soft data-consistency
  term.

```text
models/lpd/code/lpd_v2_final.pt   epoch 8, best clean validation W1
```

```bash
mkdir -p models/lpd/training/logs
sbatch models/lpd/training/submit_train_lpd_v2.slurm   # 8 V100s (gpu-dgx), ~22 min/epoch
```

Benchmark results against DBT, Tikhonov and the published NSK are in
`analysis/pd_recovery/results/lpd_v2/` (see `analysis/pd_recovery/README.md`).
Known limitations: trained noise-free and on `q_max = 10 1/A` only (it is out of
distribution at `q_max = 4`, the paper's setting), and not trained for
Delta PD.
