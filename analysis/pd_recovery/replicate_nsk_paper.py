"""
replicate_nsk_paper.py — Reproduce the NSK-ℓ1 results of Natan, PRA 107, 023105 (2023) with the
implementation in models/nsk/code/nsk.py, and test when sparse deconvolution (and the LPD) can resolve
pair-density-difference (ΔPD) spikes.

Cases (--case)
--------------
  i2   The paper's synthetic example (Fig. 2): excited diatomic iodine at τ = 150 fs. The ΔPD is a
       narrow ground-state depletion at 2.67 Å, a narrow bound excited state at 3.9 Å and a weak,
       broad dissociative component around 4.7 Å (shapes read off Fig. 2a; arbitrary units). The
       difference signal is simulated through the Debye forward model over 0.5 < q < 4 Å⁻¹
       (Δq = 0.1 Å⁻¹) with a Slepian window h(q), inverted by the DBT (PD0), and deconvolved with
       the NSK dictionary at ΔR ≈ 0.157 Å (M/N ≈ 5) with ε at the L-curve corner (Fig. 2c–d).
       Noise-free, thin-sample limit (the paper also models the 8 mm sample thickness). For
       contrast, the pipeline's ADMM (500 iterations) is run on the same PD0 and dictionary.

  coherent  A paper-like ΔPD for our own molecules (the four validation molecules of the
       analysis/pd_recovery/plot_lpd_predictions.py figures). The paper's experimental case (Fig. 5, CHD ring
       opening) is a large structural change of a few distances, compared with an ensemble-averaged
       simulation; our trajectory ΔPDs are single-frame thermal jitter (most distances move by
       < 0.1 Å). Here, a thermal ensemble (--n-ens random-phase snapshots of the pipeline's
       harmonic model at 300 K, processing/analyse_mol_v2.generate_trajectory_batch) is displaced
       coherently along one normal mode — the one that changes pair distances the most — by an
       amplitude a (largest atomic displacement, swept over --amps), and
            Δp = ⟨p⟩_displaced − ⟨p⟩_ensemble,   ΔS = ⟨S0⟩_displaced − ⟨S0⟩_ensemble
       with the true atomic form factors (the same Debye code as the dataset). Methods on the
       processed difference signal (q-range 0–10 Å⁻¹, or a beamstop with --qmin):
            NSK-ℓ1 (paper)          signed, ΔR = 0.1 Å, ε at the L-curve corner (models/nsk/code/nsk.py)
            NSK-ℓ1 (pipeline ADMM)  0.025 Å dictionary, 500 iterations, λ = 1e-4 (harness value)
            LPD v2                  v2(displaced) − v2(ensemble); never trained on Δ, and trained
                                    only on single near-equilibrium frames
            DBT                     with the scale k fitted to this Δp (oracle scale)
       Scored by R² after blurring prediction and truth by the paper's minimum separation
       (FWHM 0.15 Å, σ = 0.064 Å) and at σ = 0.025 Å, versus a.

  lpd-delta  Why LPD's ΔPD is poor on the trajectory data: ΔPD = v2(frame t) − v2(frame 0) scored
       on --n-mol training and --n-mol validation molecules (--frames each), with R² at
       σ = 0.025 / 0.064 / 0.1 Å and the amplitude ratio — the least-squares slope of the blurred
       prediction on the blurred truth (1 = full amplitude, 0 = no response to the change). The
       same numbers for the absolute PD of frame t are printed for comparison.

  fig3  The paper's quantitative test of accuracy under noise and super-resolution (Sec. III, Fig. 3),
       run for both NSK-ℓ1 and LPD v2. A linear three-atom system "similar to CO2" (O–C–O): outer
       atoms fixed at x = ∓2 Å (R_AC = 4 Å), the central atom at x_B, so R_AB = 2 + x_B and
       R_BC = 2 − x_B approach each other and merge at 2 Å. x_B ∈ [0, 1] Å (the paper's −1…1 repeats
       the same distances). Absolute PD, normalised (weights O–O 0.4, C–O 0.3 + 0.3).
       Measurement: total scattering S0 from the dataset's Debye code and form factors, observed for
       --fig3-qmin ≤ q ≤ q_max with Δq = 0.1 Å⁻¹, processed as in the pipeline (S0 − Σf²)/fe · Lorch.
       Runs once per q_max in --fig3-qmax: 4 Å⁻¹ is the paper's configuration (diffraction limit
       2π/q_max = 1.57 Å) and is OUTSIDE LPD v2's training (always q_max = 10); 10 Å⁻¹ is the
       range LPD v2 was trained for (q_min = 0.5 is inside its training beamstops 0.3–1.0).
       Noise (paper: "q-dependent additive white Gaussian noise, where each q_i bin has sampling
       statistics proportional to the number of detector pixels contributing to it", ePix10k
       detector, SNR per single detector pixel, 20 realisations per SNR). Implemented as: white
       Gaussian noise per pixel with σ_pixel² = ⟨S0²⟩ over the observed q / 10^(SNR/10), averaged over
       the pixels of each q bin, σ_bin = σ_pixel / √n_pixels(q), for a flat detector normal to the
       beam with --n-pixels pixels inside a disk whose edge is at q_max, at --photon-kev (scaled for
       other q_max so the angular coverage is the same). ASSUMPTIONS not given in the paper: the
       photon energy, the circular coverage, the pixel count, and that "signal" in the per-pixel SNR
       is the mean square total scattering. They shift the absolute dB scale (not the comparison
       between methods, which see identical noisy signals), hence the wider default --snrs.
       Sample thickness and other distortions (Fig. 1) are not modelled.
       Methods: NSK-ℓ1 as in the paper (ΔR = --fig3-dr = 0.05 Å, exact ℓ1 path, ε at the L-curve
       corner per realisation; non-negative, as the target is an absolute PD) and LPD v2 (trained
       noise-free: every noisy level is out of distribution for it), on the same processed signal.
       Scoring, following Fig. 3: peak positions of the recovered PD (recovered_distances) versus the
       exact distances. Recovery error of R_AC (Fig. 3d); error of the approaching pair, the mean of
       |R̂ − R| over R_AB and R_BC after sorting (Fig. 3e; a single merged peak counts for both, the
       paper's coalescence), versus separation |R_AB − R_BC| and SNR; and the minimum separation per
       SNR — the separation above which that error stays ≤ separation/4, the paper's definition
       (min_separation). Paper's result for NSK: δR ≈ 0.35 ± 0.15 Å over −15…30 dB at q_max = 4, and
       R_AC error < 0.1 Å. Outputs analysis/pd_recovery/results/lpd_v2/figures/nsk_fig3_qmax<q>.png and
       analysis/pd_recovery/results/lpd_v2/nsk_fig3.npz. A SLURM job (analysis/pd_recovery/submit_nsk_fig3.slurm): ≈ 28k NSK paths.

Outputs: analysis/pd_recovery/results/lpd_v2/figures/nsk_replication_<case>.png (nsk_fig3_*.png for fig3) and a printed summary.

Usage
-----
    python analysis/pd_recovery/replicate_nsk_paper.py --case i2
    python analysis/pd_recovery/replicate_nsk_paper.py --case coherent [--qmin 0.8]
    python analysis/pd_recovery/replicate_nsk_paper.py --case lpd-delta
    sbatch analysis/pd_recovery/submit_nsk_fig3.slurm            # --case fig3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal.windows import dpss
from scipy.special import spherical_jn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.nsk.code.nsk import NSK, admm_lasso  # noqa: E402


def gaussian_bins(R, centre, sigma, area):
    g = np.exp(-0.5 * ((R - centre) / sigma) ** 2)
    return area * g / g.sum()


def pd0_lam_max(nsk, y):
    """ε_max of the signed problem for PD0 vector y (the ε above which w = 0)."""
    return float(np.abs(nsk.Dn.T @ y).max())


def case_i2(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Measurement (paper Sec. II C): Δq = 0.1 Å⁻¹, 0.5 < q < 4 Å⁻¹, Slepian window.
    dq = 0.1
    q = np.arange(np.finfo(float).eps, 4.0 + dq / 2, dq)
    h = np.zeros_like(q)
    obs = q >= 0.5 - 1e-9
    h[obs] = dpss(int(obs.sum()), args.slepian_nw, sym=True)
    h /= h.max()

    # Truth (Fig. 2a, arbitrary units): depletion −1 at 2.67 Å, +0.6 bound at 3.9 Å, +0.4 dissociative.
    Rf = np.arange(0.0025, 10, 0.005)                           # fine grid for the truth
    rho = (gaussian_bins(Rf, 2.67, 0.025, -1.0) + gaussian_bins(Rf, 3.90, 0.035, 0.6)
           + gaussian_bins(Rf, 4.70, 0.20, 0.4))
    x = h * (2 * spherical_jn(0, np.outer(q, Rf)) @ rho)       # Debye: Δx = h Σ 2 Δρ j0(qR)

    nsk = NSK(q, h, args.dR_dict)
    M, R = nsk.M, nsk.R
    print(f"q points {len(q)} (observed {obs.sum()}), dictionary ΔR = {nsk.dR:.4f} Å, M = {M}, "
          f"M/N = {M / obs.sum():.1f}, diffraction limit 2π/qmax = {2 * np.pi / q[-1]:.2f} Å")
    pd0 = nsk.transform(x[None])[0]
    p, info = nsk.solve(x[None], positive=False, return_curves=True)
    p, cur = p[0], info["curves"][0]
    z_admm = admm_lasso(pd0[:, None], nsk.Dn, args.admm_lam, nonneg=False)[:, 0]
    p_admm = 0.5 * z_admm / nsk.col
    print(f"L-curve ε/ε_max = {info['lam'][0]:.2e}, nonzero stems {info['nnz'][0]} of {len(R)}")

    # truth aggregated into the dictionary bins (weights comparable with the stems)
    edges = np.concatenate([[R[0] - nsk.dR / 2], R + nsk.dR / 2])
    truth_bins = np.histogram(Rf, bins=edges, weights=rho)[0]
    for name, lo, hi_ in (("depletion 2.67 Å", 2.3, 3.0), ("bound 3.9 Å", 3.6, 4.2), ("dissociative 4.2–5.5 Å", 4.2, 5.5)):
        sel = (R >= lo) & (R < hi_)
        cen = (R[sel] * p[sel]).sum() / p[sel].sum() if abs(p[sel].sum()) > 0 else np.nan
        print(f"  {name:24s} truth weight {truth_bins[sel].sum():+.3f} | NSK-ℓ1 {p[sel].sum():+.3f} "
              f"(centroid {cen:.2f} Å) | ADMM-500 {p_admm[sel].sum():+.3f}")

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    scale = nsk.dR / 0.005                                      # truth density → per dictionary bin
    a = ax[0, 0]
    a.plot(Rf, rho * scale, color="0.6", lw=2, label="Δρ (truth)")
    a.plot(nsk.R_dbt, pd0 / np.abs(pd0).max() * np.abs(rho * scale).max(), color="C0", lw=2,
           label="ΔPD0 = DBT (rescaled)")
    a.set_title("(a) truth and direct inversion", fontsize=9)
    a = ax[0, 1]
    a.plot(Rf, rho * scale, color="0.6", lw=2, label="Δρ (truth)")
    a.plot(R, p_admm, color="C2", lw=1.2, label=f"pipeline ADMM, 500 it (λ={args.admm_lam:g})")
    a.set_title("pipeline ADMM on the same PD0 / dictionary", fontsize=9)
    a = ax[1, 0]
    a.plot(Rf, rho * scale, color="0.6", lw=2, label="Δρ (truth)")
    nz = np.abs(p) > 0
    a.plot(R, p, "r:", lw=1)
    a.plot(R[nz], p[nz], "o", color="r", ms=4, label="w/2, exact ℓ1, L-curve ε")
    a.set_title("(c) NSK-ℓ1 weights (paper method)", fontsize=9)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlim(0, 7); a.set_xlabel("R (Å)"); a.axhline(0, color="0.8", lw=0.6); a.legend(fontsize=8)
    a = ax[1, 1]
    w = cur["window"]
    sc = a.scatter(cur["residual"][w], cur["norm"][w], c=np.log10(cur["lam"][w] / pd0_lam_max(nsk, pd0)),
                   s=10, cmap="jet")
    a.plot(cur["residual"][cur["k"]], cur["norm"][cur["k"]], "k*", ms=12, label="selected corner (knee)")
    a.set_xlabel("residual norm ‖PD0 − Dw‖"); a.set_ylabel("solution norm ‖w‖₁"); a.legend(fontsize=8)
    fig.colorbar(sc, ax=a, label="log10 ε/ε_max")
    a.set_title("(d) L-curve (linear axes, low-residual part, as in the paper)", fontsize=9)
    fig.suptitle("Replication of Natan PRA 107, 023105, Fig. 2 (iodine ΔPD, 0.5 < q < 4 Å⁻¹, noise-free)", fontsize=10)
    fig.tight_layout()
    out = ROOT / "analysis/pd_recovery/results/lpd_v2/figures/nsk_replication_i2.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


SIG_MINSEP = 0.15 / (2 * np.sqrt(2 * np.log(2)))              # paper's minimum separation as FWHM → σ


def r2_blur(pred, tgt, sigma, dR):
    """R² after blurring both by a Gaussian of σ [Å] (last axis); also the amplitude slope."""
    from scipy.ndimage import gaussian_filter1d
    pb = gaussian_filter1d(pred, sigma / dR, axis=-1, mode="constant") if sigma > 0 else pred
    tb = gaussian_filter1d(tgt, sigma / dR, axis=-1, mode="constant") if sigma > 0 else tgt
    sst = ((tb - tb.mean(-1, keepdims=True)) ** 2).sum(-1)
    r2 = 1 - ((pb - tb) ** 2).sum(-1) / np.where(sst > 0, sst, np.nan)
    slope = (pb * tb).sum(-1) / np.where((tb ** 2).sum(-1) > 0, (tb ** 2).sum(-1), np.nan)
    return r2, slope


def load_v2(path):
    import torch
    from models.lpd.code.learned_primal_dual_v2 import LearnedPrimalDualV2
    ck = torch.load(path, map_location="cpu")
    model = LearnedPrimalDualV2(ck["q"], ck["R_grid"], **ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def v2_predict(model, S0, ss, fe, q, qmin):
    """LPD v2 on raw S0 rows (n, nq) processed exactly as in training. Returns (p̂ (n, nR), dS0, W)."""
    import torch
    from models.lpd.code.learned_primal_dual_v2 import process_scattering
    t = lambda a: torch.tensor(np.atleast_2d(a), dtype=torch.float32)
    S0t = t(S0)
    n = S0t.shape[0]
    dS0, W = process_scattering(S0t, t(ss).expand(n, -1), t(fe).expand(n, -1), t(q)[0],
                                torch.full((n,), float(qmin)))
    with torch.no_grad():
        p, _ = model(dS0, W)
    return p.numpy().astype(np.float64), dS0.numpy().astype(np.float64), W.numpy().astype(np.float64)


def case_coherent(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from processing.analyse_mol_v2 import (build_scattering_context, compute_ground_truth_pd_batch,
                                           generate_trajectory_batch, get_atom_names, simulate_scattering_batch)
    from models.lpd.code.learned_primal_dual import _dbt
    from models.nsk.code.nsk import NSK, admm_lasso

    amps = [float(a) for a in args.amps.split(",")]
    mols = [int(m) for m in args.mols.split(",")]
    model = load_v2(ROOT / args.checkpoint_v2)
    ref = np.load(ROOT / f"data/trajectory_data_cap200/mol_{mols[0]:06d}_pd.npz")
    q, Rg = ref["q"].astype(np.float64), ref["R_grid"].astype(np.float64)
    dR = float(Rg[1] - Rg[0])
    Rc = Rg + dR / 2
    nsk_p = nsk_f = None
    results = {}
    for mol in mols:
        ph = np.load(ROOT / f"data/mol_data/mol_{mol:06d}_physics.npz")
        names = get_atom_names(mol)
        R0, m = ph["R_0"].ravel().astype(float), ph["masses_amu"].astype(float)
        V, om = ph["V_vib"].astype(float), ph["omegas_ASE"].astype(float)
        N = len(names)
        ctx = build_scattering_context(names, q)
        pairs = ctx["all_pairs"]
        # thermal ensemble: random-phase snapshots of the dataset's harmonic model
        _, ens = generate_trajectory_batch(R0, m, V, om, T_kelvin=300.0, n_traj=args.n_ens,
                                           points_per_fast_cycle=10, seed=10_000 + mol, max_frames=1)
        # Cartesian displacement of each mode, scaled to a largest atomic displacement of 1 Å; pick the
        # mode with the largest RMS first-order change of the pair distances
        X0 = R0.reshape(N, 3)
        rab = X0[pairs[:, 1]] - X0[pairs[:, 0]]
        uab = rab / np.linalg.norm(rab, axis=1, keepdims=True)
        best, best_rms = None, -1.0
        for k in range(V.shape[1]):
            d = (V[:, k] / np.sqrt(m)).reshape(N, 3)          # masses_amu is per coordinate (3N)
            d = d / np.linalg.norm(d, axis=1).max()
            rms = np.sqrt(np.mean(((d[pairs[:, 1]] - d[pairs[:, 0]]) * uab).sum(1) ** 2))
            if rms > best_rms:
                best, best_rms, kbest = d, rms, k
        geoms = np.concatenate([ens] + [ens + a * best[None, None] for a in amps], axis=1)  # (n_ens, 1+nA, N, 3)
        S0, r_ab = simulate_scattering_batch(q, geoms, ctx, frame_batch=8)
        pd = compute_ground_truth_pd_batch(r_ab, ctx["pair_weights"], Rg)
        p = (pd / pd.sum(-1, keepdims=True)).mean(0)                                     # (1+nA, nR)
        S = S0.mean(0)                                                                   # (1+nA, nq)
        tgt = p[1:] - p[0]
        pv2, dS0, W = v2_predict(model, S, ctx["self_scatter"], ctx["fe"], q, args.qmin)
        x = dS0[1:] - dS0[0]
        if nsk_p is None:                                     # same window for every molecule
            nsk_p = NSK(q, W[0], 0.1)
            nsk_f = NSK(q, W[0], dR)
        stems, info = nsk_p.solve(x, positive=False)
        preds = {"NSK-ℓ1 (paper)": nsk_p.to_grid(stems, Rc),
                 "NSK-ℓ1 (pipeline ADMM)": nsk_f.to_grid(0.5 * admm_lasso(nsk_f.transform(x).T, nsk_f.Dn, 1e-4, False).T
                                                         / nsk_f.col, Rc),
                 "LPD v2": pv2[1:] - pv2[0]}
        y = _dbt(0, x.T, q, len(Rg) + 1)[0].T
        k_or = (y * tgt).sum() / (y * y).sum()
        preds["DBT (oracle scale)"] = k_or * y
        res = {meth: {f"R2@{s:g}": r2_blur(v, tgt, s, dR)[0] for s in (0.025, SIG_MINSEP)} for meth, v in preds.items()}
        results[mol] = dict(N=N, mode=kbest, rms=best_rms, tgt=tgt, preds=preds, stems=stems, res=res, info=info)
        print(f"\nmol {mol} ({N} atoms): mode {kbest} (ω = {om[kbest]:.2f}), RMS Δr per Å of displacement {best_rms:.2f}")
        print("  a [Å]  " + "  ".join(f"{a:>6g}" for a in amps) + "      (R² at σ = 0.064 Å / 0.025 Å)")
        for meth in preds:
            print(f"  {meth:24s}" + "  ".join(f"{v1:+.2f}/{v2:+.2f}" for v1, v2 in
                                               zip(res[meth][f"R2@{SIG_MINSEP:g}"], res[meth]["R2@0.025"])))
        print(f"  NSK paper ε/ε_max: {', '.join(f'{v:.0e}' for v in info['lam'])}; stems ≠ 0: {info['nnz']}")

    # figure: top row — ΔPD at the shown amplitude (blurred by the min. separation, NSK stems on top);
    # bottom row — R² at σ = 0.064 Å vs amplitude
    from scipy.ndimage import gaussian_filter1d
    ia = int(np.argmin(np.abs(np.array(amps) - args.show_amp)))
    colors = {"NSK-ℓ1 (paper)": "C4", "NSK-ℓ1 (pipeline ADMM)": "C2", "LPD v2": "C3", "DBT (oracle scale)": "C0"}
    fig, ax = plt.subplots(2, len(mols), figsize=(4.2 * len(mols), 7))
    for j, mol in enumerate(mols):
        r = results[mol]
        a = ax[0, j]
        bl = lambda v: gaussian_filter1d(v, SIG_MINSEP / dR, mode="constant")
        a.plot(Rc, bl(r["tgt"][ia]), "k", lw=1.2, label="truth")
        for meth, v in r["preds"].items():
            a.plot(Rc, bl(v[ia]), color=colors[meth], lw=1.0, label=meth)
        s = r["stems"][ia]
        nz = np.abs(s) > 1e-12 * max(np.abs(s).max(), 1e-300)
        sc = np.abs(bl(r["tgt"][ia])).max() / max(np.abs(s).max(), 1e-300)
        a.vlines(nsk_p.R[nz], 0, s[nz] * sc, color="C4", lw=0.6, alpha=0.6)
        a.axhline(0, color="0.7", lw=0.5)
        a.set_xlim(0, 8 if r["N"] < 20 else 10)
        a.set_title(f"mol {mol} ({r['N']} atoms), a = {amps[ia]:g} Å", fontsize=9)
        a.set_xlabel("R (Å)")
        if j == 0:
            a.set_ylabel("Δp, blurred by FWHM 0.15 Å")
            a.legend(fontsize=7)
        b = ax[1, j]
        for meth in r["preds"]:
            b.plot(amps, r["res"][meth][f"R2@{SIG_MINSEP:g}"], "o-", color=colors[meth], ms=3, label=meth)
        b.set_xscale("log"); b.set_ylim(-0.5, 1.02); b.axhline(0, color="0.7", lw=0.5)
        b.set_xlabel("coherent amplitude a (Å)"); b.grid(alpha=0.3)
        if j == 0:
            b.set_ylabel("R² (σ = 0.064 Å)")
    cond = "0 ≤ q ≤ 10 Å⁻¹" if args.qmin <= 0 else f"{args.qmin:g} ≤ q ≤ 10 Å⁻¹"
    fig.suptitle(f"Coherent mode displacement, thermal-ensemble-averaged ΔPD ({args.n_ens} snapshots), {cond}, "
                 f"noise-free. Top: a = {amps[ia]:g} Å (stems: NSK-ℓ1 weights, rescaled)", fontsize=10)
    fig.tight_layout()
    out = ROOT / f"analysis/pd_recovery/results/lpd_v2/figures/nsk_replication_coherent_qmin{args.qmin:g}.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


def case_lpd_delta(args):
    from models.lpd.dataset.dataset_lpd_v2 import split_files

    model = load_v2(ROOT / args.checkpoint_v2)
    val_files, train_files = split_files(ROOT / "data/trajectory_data_cap200", ROOT / "data/splits/split_lpd.npy")
    rng = np.random.default_rng(0)
    sig = (0.025, SIG_MINSEP, 0.1)
    for pool, files in (("train", train_files), ("val", val_files)):
        pick = rng.choice(len(files), size=min(args.n_mol, len(files)), replace=False)
        rows = {"Δp": {s: [] for s in sig}, "p": {s: [] for s in sig}}
        slope = {"Δp": {s: [] for s in sig}, "p": {s: [] for s in sig}}
        for i in pick:
            d = np.load(files[i])
            S0, pd = d["S0"], d["pd_true"]
            nt, nf, _ = S0.shape
            flat = rng.choice(nt * (nf - 1), size=min(args.frames, nt * (nf - 1)), replace=False)
            ti, fi = flat // (nf - 1), 1 + flat % (nf - 1)
            S = np.concatenate([S0[ti, fi], S0[ti, 0]]).astype(np.float64)
            pt = pd[ti, fi].astype(np.float64)
            p0 = pd[ti, 0].astype(np.float64)
            pt, p0 = pt / pt.sum(-1, keepdims=True), p0 / p0.sum(-1, keepdims=True)
            pv, _, _ = v2_predict(model, S, d["self_scatter"].astype(np.float64), d["fe"].astype(np.float64),
                                  d["q"].astype(np.float64), 0.0)
            n = len(ti)
            dR = float(d["R_grid"][1] - d["R_grid"][0])
            for s in sig:
                for key, pr, tg in (("Δp", pv[:n] - pv[n:], pt - p0), ("p", pv[:n], pt)):
                    r2, sl = r2_blur(pr, tg, s, dR)
                    rows[key][s].extend(r2.tolist())
                    slope[key][s].extend(sl.tolist())
        print(f"\n{pool}: {len(pick)} molecules × {args.frames} frames (LPD v2, {Path(args.checkpoint_v2).name})")
        for key in ("Δp", "p"):
            print(f"  {key:3s} median R²  " + "  ".join(f"σ={s:.3f}: {np.nanmedian(rows[key][s]):+.3f}" for s in sig)
                  + "   | amplitude slope " + "  ".join(f"{np.nanmedian(slope[key][s]):.2f}" for s in sig))


# ═════════════════════════════════════════════════════════════════════════════
# fig3: the paper's noise / super-resolution test (Sec. III, Fig. 3)
# ═════════════════════════════════════════════════════════════════════════════

HC_KEV_A = 12.39842                                           # hc [keV·Å]: λ = HC_KEV_A / E


def lorch_band(q, qmin, qmax):
    """Lorch window over the observed points qmin ≤ q ≤ qmax (unit sum) and the observed mask — the
    pipeline's analysis window (as models/learned_primal_dual_v2.lorch_window_batch, plus a q_max)."""
    obs = (q >= qmin - 1e-6) & (q <= qmax + 1e-6)
    W = np.zeros_like(q)
    W[obs] = np.sin(np.linspace(0, np.pi, int(obs.sum())))
    return W / W.sum(), obs


def pixels_per_bin(q, obs, dq, lam, n_pixels):
    """
    Detector pixels contributing to each q bin [q − Δq/2, q + Δq/2] for a flat detector normal to the
    beam whose (circular) edge is at the last observed bin edge, holding n_pixels pixels in total.
    A ring of scattering angle 2θ sits at radius r = z·tan 2θ, so the pixels in a bin are ∝ the ring
    area ∝ tan²2θ_hi − tan²2θ_lo; the sample–detector distance z cancels. Bins outside obs get 0.
    """
    def tan2(qq):
        th2 = 2 * np.arcsin(np.clip(qq * lam / (4 * np.pi), 0, 1))
        return np.tan(th2) ** 2
    lo, hi = np.clip(q - dq / 2, 0, None), q + dq / 2
    n = n_pixels * (tan2(hi) - tan2(lo)) / tan2(q[obs][-1] + dq / 2)
    return np.where(obs, n, 0.0)


def recovered_distances(p, Rc, dR, sigma, rel_prom, pair_window, ac_window):
    """
    Peak positions of recovered pair densities p (n, nR) on bin centres Rc — the quantity the paper
    scores. p is blurred by σ (merges an ℓ1 stem split over adjacent bins), peaks with prominence ≥
    rel_prom × the row maximum are found, and each position is refined by a parabola through the peak
    bin and its neighbours. Returns pair (n, 2): the two strongest peaks inside pair_window, sorted
    (one peak → both distances at it, i.e. coalescence; none → NaN), and ac (n,): the strongest peak
    inside ac_window.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    pb = gaussian_filter1d(p, sigma / dR, axis=-1, mode="constant") if sigma > 0 else p
    pair = np.full((len(p), 2), np.nan)
    ac = np.full(len(p), np.nan)
    for i, row in enumerate(pb):
        if row.max() <= 0:
            continue
        pk, _ = find_peaks(row, prominence=rel_prom * row.max())
        pk = pk[(pk > 0) & (pk < len(row) - 1)]
        if len(pk) == 0:
            continue
        y0, y1, y2 = row[pk - 1], row[pk], row[pk + 1]
        den = y0 - 2 * y1 + y2
        off = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0)
        pos, hgt = Rc[pk] + np.clip(off, -0.5, 0.5) * dR, y1
        sel = (pos >= pair_window[0]) & (pos < pair_window[1])
        if sel.any():
            top = np.sort(pos[sel][np.argsort(hgt[sel])[::-1][:2]])
            pair[i] = top if len(top) == 2 else (top[0], top[0])
        sel = (pos >= ac_window[0]) & (pos < ac_window[1])
        if sel.any():
            ac[i] = pos[sel][np.argmax(hgt[sel])]
    return pair, ac


_NSK_WORKER = {}


def _nsk_rows(x):
    """Worker: paper NSK-ℓ1 (non-negative, L-curve ε) for rows x — module global set before forking."""
    nsk = _NSK_WORKER["nsk"]
    stems, info = nsk.solve(x, positive=True)
    return nsk.to_grid(stems, _NSK_WORKER["Rc"]), np.array(info["lam"])


def case_fig3(args):
    """See the module docstring (--case fig3)."""
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor
    import torch
    from processing.analyse_mol_v2 import (build_scattering_context, compute_ground_truth_pd_batch,
                                           simulate_scattering_batch)

    t0 = time.time()
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    torch.set_num_threads(workers)
    model = load_v2(ROOT / args.checkpoint_v2)
    ref = np.load(ROOT / "data/trajectory_data_cap200/mol_021791_pd.npz")   # the dataset's q / R grids
    q, Rg = ref["q"].astype(np.float64), ref["R_grid"].astype(np.float64)
    dq, dR = float(q[1] - q[0]), float(Rg[1] - Rg[0])
    Rc = Rg + dR / 2

    # three-atom O–C–O system: A, C fixed at x = ∓2 Å, B at x_B ∈ [0, 1] Å (x_B < 0 gives the same
    # distances with R_AB ↔ R_BC swapped); separation |R_AB − R_BC| = 2 x_B
    xb = np.round(np.arange(0, 1 + 1e-9, args.xb_step), 6)
    names = ["O", "C", "O"]
    geoms = np.zeros((1, len(xb), 3, 3))
    geoms[0, :, 0, 0], geoms[0, :, 2, 0], geoms[0, :, 1, 0] = -2.0, 2.0, xb
    ctx = build_scattering_context(names, q)
    S0, r_ab = simulate_scattering_batch(q, geoms, ctx)
    S0, r_ab = S0[0], r_ab[0]                                  # (n_pos, nq), (n_pos, 3) pairs (AB, AC, BC)
    p_true = compute_ground_truth_pd_batch(r_ab[None], ctx["pair_weights"], Rg)[0].astype(np.float64)
    p_true /= p_true.sum(-1, keepdims=True)
    pairs = ctx["all_pairs"].tolist()
    i_ab, i_ac, i_bc = pairs.index([0, 1]), pairs.index([0, 2]), pairs.index([1, 2])
    true_pair = np.sort(r_ab[:, [i_ab, i_bc]], axis=1)
    true_ac = r_ab[:, i_ac]
    sep = true_pair[:, 1] - true_pair[:, 0]

    snrs = [float(s) for s in args.snrs.split(",")]
    levels = snrs + [np.inf]                                    # + the noise-free reference
    qmaxes = [float(v) for v in args.fig3_qmax.split(",")]
    lam_ref = HC_KEV_A / args.photon_kev
    q_edge_ref = args.paper_qmax + dq / 2
    out = {}
    for qmax in qmaxes:
        W, obs = lorch_band(q, args.fig3_qmin, qmax)
        # same detector (same angular coverage) for every q_max: the photon energy is scaled so that the
        # detector edge sits at q_max (for q_max = paper_qmax it is --photon-kev)
        lam = lam_ref * q_edge_ref / (qmax + dq / 2)
        npix = pixels_per_bin(q, obs, dq, lam, args.n_pixels)
        nsk = NSK(q, W, args.fig3_dr)
        _NSK_WORKER.update(nsk=nsk, Rc=Rc)
        print(f"\n=== q ∈ [{args.fig3_qmin:g}, {qmax:g}] Å⁻¹ ({obs.sum()} bins; diffraction limit 2π/q_max = "
              f"{2 * np.pi / qmax:.2f} Å), photon energy {HC_KEV_A / lam:.1f} keV, pixels per bin "
              f"{npix[obs].min():.0f}–{npix[obs].max():.0f}; NSK ΔR = {nsk.dR:.4f} Å ({len(nsk.R)} kernels, "
              f"M/N = {nsk.M / obs.sum():.1f})", flush=True)
        rng = np.random.default_rng(args.seed)
        res = dict(err_pair={}, err_ac={}, fail={}, show={}, eps={})
        n_rep_all = []
        for lev in levels:
            n_rep = 1 if np.isinf(lev) else args.n_real
            S = np.repeat(S0[None], n_rep, 0)                                     # (n_rep, n_pos, nq)
            if not np.isinf(lev):
                # per-pixel SNR on the total scattering: σ_pixel² = ⟨S0²⟩_observed / 10^(SNR/10); a bin
                # averages npix pixels → σ_bin = σ_pixel / √npix (white per pixel, q-dependent per bin)
                sig_pix = np.sqrt((S0[:, obs] ** 2).mean(-1) / 10 ** (lev / 10))   # (n_pos,)
                sig_bin = sig_pix[:, None] / np.sqrt(np.where(obs, npix, 1.0))[None]
                S = S + rng.normal(size=S.shape) * sig_bin[None] * obs
            S = S.reshape(-1, len(q))
            x = (S - ctx["self_scatter"]) / ctx["fe"] * W                         # processed signal
            # LPD v2 — the same processed signal and window
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float32)
                p_lpd = np.concatenate([model(xt[i:i + 512], torch.tensor(W, dtype=torch.float32)
                                              .expand(len(xt[i:i + 512]), -1))[0].numpy()
                                        for i in range(0, len(xt), 512)]).astype(np.float64)
            # NSK — rows in parallel
            chunks = np.array_split(x, max(1, min(len(x), 4 * workers)))
            if workers > 1:
                with ProcessPoolExecutor(workers) as ex:
                    parts = list(ex.map(_nsk_rows, chunks))
            else:
                parts = [_nsk_rows(c) for c in chunks]
            p_nsk = np.concatenate([a for a, _ in parts])
            res["eps"][lev] = np.concatenate([b for _, b in parts])
            for meth, p in (("NSK-ℓ1 (paper)", p_nsk), ("LPD v2", p_lpd)):
                pr, ac = recovered_distances(p, Rc, dR, args.peak_sigma, args.peak_prom, (0.7, 3.4), (3.4, 4.8))
                pr, ac = pr.reshape(n_rep, len(xb), 2), ac.reshape(n_rep, len(xb))
                e_pair = np.abs(pr - true_pair[None]).mean(-1)                     # (n_rep, n_pos)
                res["err_pair"].setdefault(meth, {})[lev] = e_pair
                res["err_ac"].setdefault(meth, {})[lev] = np.abs(ac - true_ac[None])
                res["fail"].setdefault(meth, {})[lev] = np.isnan(e_pair).mean(0)
                res["show"].setdefault(meth, {})[lev] = p.reshape(n_rep, len(xb), -1)[0]
            print(f"  SNR {lev:>5g} dB: {n_rep * len(xb)} inversions  ({time.time() - t0:.0f} s)", flush=True)
            n_rep_all.append(n_rep)
        out[qmax] = res

        # summary: minimum separation (paper's definition: the separation above which the average error
        # of the two approaching distances stays ≤ 1/4 of the separation — half the distance between each
        # pair and their joint centre) and the R_AC error, per SNR
        print(f"  {'SNR [dB]':>9} | " + " | ".join(f"{m:^31s}" for m in res["err_pair"]))
        print(f"  {'':>9} | " + " | ".join(f"{'δR_min [Å]':>10} {'R_AC err [Å]':>12} {'fail':>6}"
                                          for _ in res["err_pair"]))
        for lev in levels:
            row = []
            for meth in res["err_pair"]:
                ms = min_separation(res["err_pair"][meth][lev], sep)
                row.append(f"{ms:10.3f} {np.nanmean(res['err_ac'][meth][lev]):12.3f} "
                           f"{res['fail'][meth][lev].mean():6.2f}")
            print(f"  {lev:>9g} | " + " | ".join(row))
        print(f"  NSK L-curve ε/ε_max median per level: "
              + ", ".join(f"{lev:g}: {np.median(res['eps'][lev]):.1e}" for lev in levels))

    plot_fig3(args, out, xb, sep, Rc, p_true, levels)
    np.savez_compressed(ROOT / "analysis/pd_recovery/results/lpd_v2/nsk_fig3.npz", xb=xb, sep=sep, levels=np.array(levels),
                        qmax=np.array(qmaxes),
                        **{f"q{qm:g}_{k}_{m.split()[0]}_{lev:g}": v
                           for qm, r in out.items() for k in ("err_pair", "err_ac")
                           for m, d in r[k].items() for lev, v in d.items()})
    print(f"Wrote analysis/pd_recovery/results/lpd_v2/nsk_fig3.npz  (total {time.time() - t0:.0f} s)")


def min_separation(err, sep):
    """
    Paper's minimum separation from the errors err (n_rep, n_pos) of the two approaching distances:
    the smallest separation s* such that the average error ≤ s/4 for every separation s ≥ s*
    (s/4 = half the distance between each pair and their joint centre of mass). Positions where most
    realisations found no peak count as unresolved. NaN if not resolved even at the largest separation.
    """
    with np.errstate(invalid="ignore"):
        e = np.nanmean(err, 0)
        ok = (e <= sep / 4) & (np.isnan(err).mean(0) < 0.5) & (sep > 0)
    order = np.argsort(sep)
    ok, s = ok[order], sep[order]
    if not ok[-1]:
        return np.nan
    bad = np.nonzero(~ok)[0]
    return float(s[bad[-1] + 1]) if len(bad) else float(s[ok][0])


def plot_fig3(args, out, xb, sep, Rc, p_true, levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [float(s) for s in args.show_snrs.split(",")]
    snr_f = [l for l in levels if np.isfinite(l)]
    for qmax, res in out.items():
        meths = list(res["err_pair"])
        fig, ax = plt.subplots(3, len(show) + 1, figsize=(3.3 * (len(show) + 1), 9.5))
        for i, meth in enumerate(meths):
            for j, lev in enumerate(show):
                a = ax[i, j]
                img = res["show"][meth][lev]
                sel = (Rc > 0.5) & (Rc < 4.6)
                a.imshow(img[:, sel].T, origin="lower", aspect="auto", cmap="viridis",
                         extent=[xb[0], xb[-1], Rc[sel][0], Rc[sel][-1]],
                         vmax=np.percentile(img[:, sel], 99.5) if img[:, sel].max() > 0 else None)
                a.plot(xb, 2 + xb, "r:", lw=0.8)
                a.plot(xb, 2 - xb, "r:", lw=0.8)
                a.axhline(4, color="r", ls=":", lw=0.8)
                a.set_title(f"{meth}, {lev:g} dB", fontsize=9)
                a.set_xlabel("x_B (Å)")
                if j == 0:
                    a.set_ylabel("R (Å)")
            a = ax[i, -1]
            E = np.array([np.nanmean(res["err_pair"][meth][l], 0) for l in snr_f])     # (n_snr, n_pos)
            order = np.argsort(sep)
            im = a.pcolormesh(snr_f, sep[order], E[:, order].T, cmap="Blues_r", vmin=0, vmax=0.25, shading="nearest")
            a.plot(snr_f, [min_separation(res["err_pair"][meth][l], sep) for l in snr_f], "r-", lw=1.5,
                   label="min. separation")
            a.set_xlabel("SNR per pixel (dB)")
            a.set_ylabel("|R_AB − R_BC| (Å)")
            a.set_title(f"{meth}: error of R_AB, R_BC (Å)", fontsize=9)
            a.legend(fontsize=7, loc="upper right")
            fig.colorbar(im, ax=a)
        colors = {"NSK-ℓ1 (paper)": "C4", "LPD v2": "C3"}
        a = ax[2, 0]
        for meth in meths:
            m = np.array([np.nanmean(res["err_ac"][meth][l]) for l in snr_f])
            s = np.array([np.nanstd(res["err_ac"][meth][l]) for l in snr_f])
            a.plot(snr_f, m, color=colors[meth], label=meth)
            a.fill_between(snr_f, np.maximum(m - s, 0), m + s, color=colors[meth], alpha=0.2)
        a.set_xlabel("SNR per pixel (dB)"); a.set_ylabel("R_AC error (Å)"); a.legend(fontsize=7)
        a.set_title("fixed distance R_AC = 4 Å", fontsize=9)
        a = ax[2, 1]
        for meth in meths:
            a.plot(snr_f, [min_separation(res["err_pair"][meth][l], sep) for l in snr_f], "o-", ms=3,
                   color=colors[meth], label=meth)
        a.axhline(2 * np.pi / qmax, color="0.5", ls="--", lw=0.8, label="diffraction limit 2π/q_max")
        a.set_xlabel("SNR per pixel (dB)"); a.set_ylabel("minimum separation (Å)"); a.legend(fontsize=7)
        a.set_title("resolvable separation", fontsize=9)
        a = ax[2, 2]
        for meth in meths:
            for sv, ls in ((0.4, "-"), (1.0, "--")):
                k = int(np.argmin(np.abs(sep - sv)))
                a.plot(snr_f, [np.nanmean(res["err_pair"][meth][l][:, k]) for l in snr_f], ls,
                       color=colors[meth], label=f"{meth}, sep {sep[k]:.2f} Å")
        a.set_xlabel("SNR per pixel (dB)"); a.set_ylabel("R_AB, R_BC error (Å)"); a.legend(fontsize=7)
        a.set_title("approaching pair at fixed separation", fontsize=9)
        for a in ax[2, 3:]:
            a.axis("off")
        fig.suptitle(f"Natan PRA 107, 023105, Fig. 3 test (O–C–O, R_AC = 4 Å), q ∈ [{args.fig3_qmin:g}, {qmax:g}] Å⁻¹; "
                     f"rows 1–2: recovered PD (one noise draw; red: true distances)", fontsize=10)
        fig.tight_layout()
        path = ROOT / f"analysis/pd_recovery/results/lpd_v2/figures/nsk_fig3_qmax{qmax:g}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="i2", choices=("i2", "coherent", "lpd-delta", "fig3"))
    ap.add_argument("--dR-dict", type=float, default=0.157, help="i2: NSK dictionary resolution [Å] (paper: 0.157)")
    ap.add_argument("--slepian-nw", type=float, default=2.0, help="i2: time-half-bandwidth of the Slepian window")
    ap.add_argument("--admm-lam", type=float, default=1e-4, help="i2: λ fraction for the pipeline-ADMM contrast")
    ap.add_argument("--mols", default="21791,30292,8604,45129",
                    help="coherent: molecule indices (default: the plot_lpd_predictions validation panels)")
    ap.add_argument("--amps", default="0.02,0.05,0.1,0.2,0.3,0.5",
                    help="coherent: displacement amplitudes [Å] (largest atomic displacement)")
    ap.add_argument("--show-amp", type=float, default=0.3, help="coherent: amplitude shown in the top row")
    ap.add_argument("--n-ens", type=int, default=100, help="coherent: thermal snapshots per ensemble")
    ap.add_argument("--qmin", type=float, default=0.0, help="coherent: beamstop edge [Å⁻¹]")
    ap.add_argument("--n-mol", type=int, default=12, help="lpd-delta: molecules per pool")
    ap.add_argument("--frames", type=int, default=20, help="lpd-delta: frames per molecule")
    ap.add_argument("--fig3-qmin", type=float, default=0.5, help="fig3: lowest observed q [Å⁻¹] (paper: 0.5)")
    ap.add_argument("--fig3-qmax", default="4,10",
                    help="fig3: comma-separated highest observed q [Å⁻¹], one run each (paper: 4; LPD v2 was "
                         "trained with q_max = 10 only)")
    ap.add_argument("--fig3-dr", type=float, default=0.05, help="fig3: NSK dictionary resolution [Å] (paper: 0.05)")
    ap.add_argument("--xb-step", type=float, default=0.025, help="fig3: step of the central-atom position x_B [Å]")
    ap.add_argument("--snrs", default="-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30",
                    help="fig3: per-pixel SNRs [dB] (paper: −15 to 30; wider here because the absolute dB "
                         "scale depends on unstated details, see the docstring)")
    ap.add_argument("--n-real", type=int, default=20, help="fig3: noise realisations per SNR (paper: 20)")
    ap.add_argument("--photon-kev", type=float, default=9.5,
                    help="fig3: photon energy [keV] for q_max = --paper-qmax; it fixes the detector's angular "
                         "coverage, kept for the other q_max by scaling the energy")
    ap.add_argument("--paper-qmax", type=float, default=4.0, help="fig3: q_max at which --photon-kev applies")
    ap.add_argument("--n-pixels", type=float, default=2.16e6,
                    help="fig3: detector pixels inside the observed disk (ePix10k2M ≈ 2.16e6)")
    ap.add_argument("--peak-sigma", type=float, default=0.025,
                    help="fig3: Gaussian σ [Å] applied before peak finding (merges a stem split over 2 bins)")
    ap.add_argument("--peak-prom", type=float, default=0.1,
                    help="fig3: minimum peak prominence as a fraction of the row maximum")
    ap.add_argument("--show-snrs", default="15,0,-15", help="fig3: SNRs [dB] of the PD-vs-x_B panels")
    ap.add_argument("--workers", type=int, default=0, help="fig3: NSK worker processes (0 = $SLURM_CPUS_PER_TASK or 1)")
    ap.add_argument("--seed", type=int, default=0, help="fig3: noise seed")
    ap.add_argument("--checkpoint-v2", default="models/lpd/code/lpd_v2_final.pt")
    args = ap.parse_args()
    {"i2": case_i2, "coherent": case_coherent, "lpd-delta": case_lpd_delta, "fig3": case_fig3}[args.case](args)
