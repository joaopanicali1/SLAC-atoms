# Analysis

Analysis is organized by model/task:

```text
beamstop/     low-q beamstop recovery benchmark and sensitivity analyses
pd_recovery/  full pair-density recovery: LPD v2 benchmark, NSK paper Fig. 3
              benchmark, and older UNet/classical comparisons
```

Final beamstop evaluation code lives in `analysis/beamstop/`.

Run the primary benchmark:

```bash
sbatch analysis/beamstop/submit_eval_beamstop_features_benchmark.slurm
```

This compares:

- beamstop conditioned
- beamstop unconditioned
- NSK baseline

Outputs are saved under:

```text
analysis/beamstop/final_results/
```

Sensitivity analyses:

```bash
sbatch analysis/beamstop/submit_eval_beamstop_qmin_sensitivity.slurm
sbatch analysis/beamstop/submit_eval_beamstop_snr_sensitivity.slurm
```

PD-recovery comparisons live in `analysis/pd_recovery/`:

```bash
python analysis/pd_recovery/evaluate_methods.py \
  --val-split data/splits/split_unet.npy \
  --traj-dir data/trajectory_data_cap200_new \
  --ckpt models/unet/code/unet_best_model_epoch208.pt
```

`analysis/pd_recovery/plot_lpd_unet_predictions.py` contains older plotting
utilities for UNet/LPD experiments.

LPD v2 benchmark (DBT, Tikhonov, published NSK-l1, LPD v1/v2):

```bash
mkdir -p analysis/logs
for i in 0 1 2; do sbatch --job-name=eval_pd_$i analysis/pd_recovery/submit_eval_pd.slurm $i; done
sbatch analysis/pd_recovery/submit_nsk_fig3.slurm
```

See `analysis/pd_recovery/README.md` for the metrics and results.
