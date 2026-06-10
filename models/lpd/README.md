# Learned Primal-Dual

Learns an unrolled DBT-based reconstruction from q-space scattering to
pair-density.

```text
code/       LearnedPrimalDual implementation
training/   training script and Slurm submit file
dataset/    LPDDataset
```

The final reusable checkpoint is:

```text
models/lpd/code/lpd_final.pt
```

Train:

```bash
sbatch models/lpd/training/submit_train_lpd.slurm
```
