# UNet

Predicts full pair density from resolution-limited pair-density input with
normal-mode auxiliary features.

```text
code/       PairDensityUNet implementation and committed checkpoints
training/   training script and Slurm submit file
dataset/    MolScatteringDataset
```

Best checkpoint:

```text
models/unet/code/unet_best_model_epoch208.pt
```

Train:

```bash
sbatch models/unet/training/submit_train_unet.slurm
```
