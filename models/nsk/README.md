# NSK-l1 (published baseline)

The natural-scattering-kernel l1 deconvolution of Natan, Phys. Rev. A 107,
023105 (2023), as published:

1. Discrete Bessel transform of the processed signal onto a coarse grid
   (spacing `dR_dict`, the paper uses about 0.1 A; `M = ceil(pi / (dR dq)) + 1`).
2. Dictionary `D[:, m] = DBT(h * j0(q R_m))`: each column is how a single sharp
   distance looks after the same q-range, window `h(q)` and inversion.
3. Exact lasso (LARS/homotopy) path, non-negative for absolute PD, signed for
   Delta PD; epsilon at the L-curve corner; `p = w / 2`.

```text
code/nsk.py   NSK class (transform, path, lcurve, weights, solve, to_grid),
              lasso_path, path_at, and admm_lasso (the pipeline's 500-iteration
              ADMM, kept only for comparison: it returns a smooth, non-sparse
              iterate instead of the lasso minimiser)
```

No training or dataset: it has no learned parameters. Used by
`analysis/pd_recovery/eval_pd_recovery.py`, `replicate_nsk_paper.py` and
`plot_lpd_predictions.py`. The beamstop benchmark keeps its own NSK baseline in
`analysis/beamstop/baseline_beamstop.py`.
