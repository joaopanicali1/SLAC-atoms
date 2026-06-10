# PD Recovery Analysis

This folder contains the older full pair-density recovery comparisons:

```text
evaluate_methods.py       raw DBT, L2 ridge, L1/TV ADMM, and UNet comparison
plot_baseline_comps.py    aggregate and per-molecule plots from saved results
results/                  saved metrics, prediction arrays, and figures
```

Typical commands:

```bash
python analysis/pd_recovery/evaluate_methods.py \
  --val-split data/splits/split_unet.npy \
  --traj-dir data/trajectory_data_cap200_new \
  --ckpt models/unet/code/unet_best_model_epoch208.pt \
  --out-dir analysis/pd_recovery/results

python analysis/pd_recovery/plot_baseline_comps.py \
  --results-dir analysis/pd_recovery/results \
  --traj-dir data/trajectory_data_cap200_new \
  --out-dir analysis/pd_recovery/results/figures
```
