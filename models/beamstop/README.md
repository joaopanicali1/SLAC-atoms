# Beamstop

Predicts the missing low-q region of normalized differential scattering curves.

```text
code/       BeamstopModel implementation and final checkpoint
training/   training script and Slurm submit file
dataset/    BeamstopDataset
```

Final model:

```text
models/beamstop/code/beamstop_final.pt
models/beamstop/code/beamstop_final.json
```

Train:

```bash
sbatch models/beamstop/training/submit_train_beamstop_features_expanded.slurm
```
