# PD Recovery Analysis

Full pair-density recovery from scattering. Two generations of comparisons live
here.

## LPD v2 benchmark

```text
eval_pd_recovery.py       harness: DBT, Tikhonov, NSK-l1 (paper; L-curve and tuned
                          epsilon), LPD v1, LPD v2 on the split_lpd.npy validation set
submit_eval_pd.slurm      one CPU job per condition (0 clean, 1 beamstop 0.8, 2 Delta PD)
replicate_nsk_paper.py    reproductions of the NSK paper (--case i2, coherent,
                          lpd-delta, fig3)
submit_nsk_fig3.slurm     the paper's Fig. 3 super-resolution/noise test (16 CPUs)
plot_lpd_predictions.py   PD panels for four molecules (LPD v2, NSK stems, DBT, v1)
results/lpd_v2/           result JSONs, nsk_fig3.npz and figures
```

All methods are scored on the normalised pair density
`p(R) = pd_true(R) / sum_{a<b} Z_a Z_b` on 0.025 A bins, the quantity the
scattering determines. Tikhonov and DBT parameters are tuned on 4 training
molecules; every metric is computed on all 66 validation molecules, 20 frames
each (1,320 frames), noise-free.

Metrics:

- **R^2 at sigma.** Prediction and truth are both blurred by a Gaussian of
  standard deviation sigma before R^2 is computed, so a peak displaced by much
  less than sigma is not punished as a miss. sigma = 0.025 / 0.05 / 0.1 A.
- **W1.** 1-D Wasserstein (earth-mover) distance in A, up to 1 A past the
  longest true pair distance.
- **junk.** Weight placed beyond that cut-off.

Medians:

| Condition | Metric | DBT | Tikhonov | NSK-l1 (L-curve) | NSK-l1 (tuned) | LPD v1 | LPD v2 |
|---|---|---|---|---|---|---|---|
| Clean | R^2 0.025 / 0.05 / 0.1 | .48 / .70 / .89 | .52 / .73 / .91 | .21 / .69 / .92 | .19 / .68 / .92 | .65 / .81 / .93 | **.74 / .90 / .98** |
| Clean | W1 [A] | .306 | .074 | .099 | .071 | .200 | **.037** |
| Beamstop 0.8 | R^2 0.025 / 0.05 / 0.1 | .06 / .08 / .05 | .22 / .34 / .47 | .16 / .61 / .87 | .17 / .62 / .88 | < 0 | **.73 / .90 / .98** |
| Beamstop 0.8 | W1 [A] | 3.98 | 2.16 | .270 | .158 | 1.40 | **.043** |

Delta PD (`PD(t) - PD(0)`) is not solved by any method yet: R^2 at 0.1 A is
0.42 (Tikhonov), 0.47-0.49 (NSK) and 0.67 (LPD v2, which was not trained for
it). Files: `results/lpd_v2/pd_eval_{clean,bs_q08,delta}.json`.

### NSK paper Fig. 3 benchmark

O-C-O with the outer atoms fixed at -/+2 A and the middle atom moved along the
axis, so the two bonded distances approach each other. Detector-pixel noise
(flat detector, 2.16e6 pixels, SNR on the total signal). The figure of merit is
the paper's minimum separation: the smallest separation above which the
recovered-peak position error stays below separation/4.

| q-range | NSK-l1 | LPD v2 |
|---|---|---|
| q <= 4 (paper) | 0.45-0.55 A at SNR >= -5 dB (paper: 0.35 +/- 0.15) | 1.85 A (out of training range) |
| q <= 10 | 0.50 A at 0 dB, 0.20-0.25 A at >= 15 dB | 0.30 A at 0 dB, 0.20 A at >= 5 dB |

The 0.2 A floor is set by the peak-finding metric. Figures:
`results/lpd_v2/figures/nsk_fig3_qmax{4,10}.png`; arrays: `results/lpd_v2/nsk_fig3.npz`.

### Commands

Run from the repository root. `data/trajectory_data_cap200` (the original
662-molecule set) must be present.

```bash
mkdir -p analysis/logs
for i in 0 1 2; do sbatch --job-name=eval_pd_$i analysis/pd_recovery/submit_eval_pd.slurm $i; done
sbatch analysis/pd_recovery/submit_nsk_fig3.slurm

PYTHONPATH=$PWD python analysis/pd_recovery/plot_lpd_predictions.py             # clean
PYTHONPATH=$PWD python analysis/pd_recovery/plot_lpd_predictions.py --qmin 0.8  # beamstop
```

Each script's docstring documents its flags, the measurement model and its
assumptions.

## Older UNet / classical comparisons

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
