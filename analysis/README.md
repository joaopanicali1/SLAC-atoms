# Analysis

Analysis is organized by model/task:

```text
beamstop/     low-q beamstop recovery benchmark and sensitivity analyses
pd_recovery/  full pair-density recovery comparisons for UNet/classical methods
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
