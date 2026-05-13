"""
benchmark_configs_analyze.py — Aggregate per-molecule .npz results and
produce comparison tables and figures.

Run after all three molecule jobs have completed:
    python processing/Benchmarking/benchmark_configs_analyze.py

Outputs
-------
  benchmark_configs_frequencies.csv   — all frequencies per mol/config
  benchmark_configs_deviations.csv    — per-mode deviations vs wB97M-V
  benchmark_configs.png               — 3×3 summary figure
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "processing", "Benchmarking", "configs_results")
OUT_DIR      = os.path.join(PROJECT_ROOT, "processing", "Benchmarking")

MOL_ORDER = ["H2O_3atoms", "CH3NO2_7atoms", "C3HAuO2_7atoms"]
CONFIGS   = ["B3LYP_RIJCOSX", "wB97M-V"]
REF       = "wB97M-V"


def mac_matrix(V1, V2):
    """Maximum Absolute Correlation: MAC[i,j] = |v1_i · v2_j|."""
    return np.abs(V1.T @ V2)


# ── Load results ──────────────────────────────────────────────────────────────
results = {}
for mol_name in MOL_ORDER:
    path = os.path.join(RESULTS_DIR, f"{mol_name}.npz")
    if not os.path.exists(path):
        print(f"  Skipping {mol_name} — results not yet available.")
        continue
    d = np.load(path, allow_pickle=True)
    available = [cfg for cfg in CONFIGS if f"freqs_{cfg}" in d.files]
    results[mol_name] = {
        cfg: (d[f"freqs_{cfg}"], d[f"V_{cfg}"]) for cfg in available
    }

mol_order_available = [m for m in MOL_ORDER if m in results]

# ── Build output tables ───────────────────────────────────────────────────────
freq_rows, dev_rows = [], []

for mol_name in mol_order_available:
    ref_freqs, ref_V = results[mol_name][REF]
    n_ref = len(ref_freqs)

    for config_name in results[mol_name]:
        freqs, V = results[mol_name][config_name]
        for k in range(min(len(freqs), n_ref)):
            freq_rows.append({
                "molecule": mol_name,
                "config":   config_name,
                "mode":     k + 1,
                "freq_cm1": round(freqs[k], 2),
            })

        if config_name != REF:
            n = min(len(freqs), n_ref)
            for k in range(n):
                delta   = freqs[k] - ref_freqs[k]
                rel_pct = delta / ref_freqs[k] * 100
                overlap = float(np.abs(V[:, k] @ ref_V[:, k]))
                dev_rows.append({
                    "molecule":  mol_name,
                    "config":    config_name,
                    "mode":      k + 1,
                    "freq_cm1":  round(freqs[k], 2),
                    "ref_cm1":   round(ref_freqs[k], 2),
                    "delta_cm1": round(delta, 3),
                    "delta_pct": round(rel_pct, 4),
                    "mac":       round(overlap, 6),
                })

df_freq = pd.DataFrame(freq_rows)
df_dev  = pd.DataFrame(dev_rows)

df_freq.to_csv(os.path.join(OUT_DIR, "benchmark_configs_frequencies.csv"), index=False)
df_dev.to_csv( os.path.join(OUT_DIR, "benchmark_configs_deviations.csv"),  index=False)

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\n── Frequency deviation summary (vs {REF}) ──")
for mol_name in mol_order_available:
    print(f"\n  {mol_name}:")
    non_ref = [c for c in results[mol_name] if c != REF]
    for config_name in non_ref:
        sub = df_dev[(df_dev.molecule == mol_name) & (df_dev.config == config_name)]
        if sub.empty:
            continue
        print(f"    {config_name:20s}  "
              f"mean |Δν| = {sub.delta_cm1.abs().mean():.2f} cm⁻¹  "
              f"max |Δν| = {sub.delta_cm1.abs().max():.2f} cm⁻¹  "
              f"mean MAC = {sub.mac.mean():.4f}")

# ── Figure: 3 rows (molecules) × 3 cols ──────────────────────────────────────
config_colors  = {"B3LYP_RIJCOSX": "steelblue", "wB97M-V": "k"}
config_labels  = {
    "B3LYP_RIJCOSX": "B3LYP-D4 RIJCOSX (production)",
    "wB97M-V":       "wB97M-V (reference)",
}

n_mols = len(mol_order_available)
fig, axes = plt.subplots(n_mols, 3, figsize=(18, 4.5 * n_mols))
if n_mols == 1:
    axes = axes[np.newaxis, :]

for row_idx, mol_name in enumerate(mol_order_available):
    ref_freqs, ref_V = results[mol_name][REF]
    n_modes   = len(ref_freqs)
    mode_idx  = np.arange(1, n_modes + 1)

    # Col 0: frequency comparison across all configs
    ax = axes[row_idx, 0]
    for config_name in results[mol_name]:
        freqs, _ = results[mol_name][config_name]
        n = min(len(freqs), n_modes)
        ax.plot(mode_idx[:n], freqs[:n],
                color=config_colors[config_name],
                lw=1.5 if config_name == REF else 1.0,
                ls="-"  if config_name == REF else "--",
                label=config_labels[config_name])
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Frequency (cm⁻¹)")
    ax.set_title(f"{mol_name}  — frequencies")
    if row_idx == 0:
        ax.legend(fontsize=7, loc="upper left")

    # Col 1: |Δν| deviation from wB97M-V
    ax = axes[row_idx, 1]
    non_ref = [c for c in results[mol_name] if c != REF]
    for config_name in non_ref:
        freqs, _ = results[mol_name][config_name]
        n = min(len(freqs), n_modes)
        ax.plot(mode_idx[:n], np.abs(freqs[:n] - ref_freqs[:n]),
                color=config_colors[config_name],
                lw=1.2, marker="o", ms=3,
                label=config_labels[config_name])
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("|Δν|  (cm⁻¹)")
    ax.set_title(f"{mol_name}  — deviation from {REF}")
    if row_idx == 0:
        ax.legend(fontsize=7)

    # Col 2: MAC matrix — B3LYP RIJCOSX vs wB97M-V mode shapes
    ax = axes[row_idx, 2]
    _, V_prod = results[mol_name]["B3LYP_RIJCOSX"]
    n = min(V_prod.shape[1], ref_V.shape[1])
    MAC = mac_matrix(V_prod[:, :n], ref_V[:, :n])
    im  = ax.imshow(MAC, vmin=0, vmax=1, cmap="viridis",
                    origin="upper", aspect="auto")
    ax.set_xlabel("wB97M-V mode index")
    ax.set_ylabel("B3LYP RIJCOSX mode index")
    ax.set_title(f"{mol_name}  — mode overlap (MAC)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(
    "Config benchmark: B3LYP-D4 RIJCOSX  |  wB97M-V (ref)\n"
    "All: def2-TZVPD / TightOpt",
    fontsize=11,
)
fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "benchmark_configs.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {os.path.join(OUT_DIR, 'benchmark_configs_frequencies.csv')}")
print(f"Saved: {os.path.join(OUT_DIR, 'benchmark_configs_deviations.csv')}")
print(f"Saved: {out_fig}")
