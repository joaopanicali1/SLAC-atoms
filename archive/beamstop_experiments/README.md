# Beamstop Experiment Archive

This folder keeps historical beamstop scripts and generated figures that are no
longer part of the current final workflow.

Current workflow entry points remain in place:

- `models/beamstop/training/train_beamstop.py`
- `models/beamstop/training/submit_train_beamstop_features_expanded.slurm`
- `analysis/evaluate_beamstop_three_way.py`
- `analysis/submit_eval_beamstop_features_benchmark.slurm`
- `analysis/baseline_beamstop.py`

Archived files are retained for reproducibility of earlier Optuna, fine-tuning,
checkpoint comparison, and baseline-only experiments.
