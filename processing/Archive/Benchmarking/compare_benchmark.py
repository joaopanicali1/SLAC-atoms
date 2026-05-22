"""
Read all benchmark CSVs and print a comparison table across node counts.
Run any time — shows "—" for results not yet available.
  python processing/compare_benchmark.py
"""
import os
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR     = os.path.join(PROJECT_ROOT, "processing")

CONFIGS = {
    "large": [
        ("ORCA 1N/32c",  "large_32cores"),
        ("ORCA 2N/64c",  "large_64cores"),
    ],
    "xlarge": [
        ("ORCA 1N/32c",   "xlarge_32cores"),
        ("ORCA 2N/64c",   "xlarge_64cores"),
        ("ORCA 4N/128c",  "xlarge_128cores"),
        ("ORCA 8N/256c",  "xlarge_256cores"),
    ],
}


def load_csv(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {int(r["mol_idx"]): r for r in csv.DictReader(f)}


def fmt_min(val):
    try:
        v = float(val)
        if v != v:   # nan
            return f"{'—':>8s}"
        return f"{v:>8.1f}"
    except (ValueError, TypeError):
        return f"{'—':>8s}"


for tier, configs in CONFIGS.items():
    labels   = [label for label, _ in configs]
    suffixes = [suffix for _, suffix in configs]
    data     = {s: load_csv(os.path.join(PROC_DIR, f"benchmark_results_{s}.csv"))
                for s in suffixes}

    all_mol_idx = sorted({i for rows in data.values() for i in rows})
    if not all_mol_idx:
        print(f"\n{tier.upper()} — no results yet\n")
        continue

    col_w   = 11
    hdr_row = "  ".join(f"{lbl:>{col_w}s}" for lbl in labels)
    print(f"\n{'='*72}")
    print(f"{tier.upper()} tier  (wall time, minutes)")
    print(f"{'Atoms':>6s}  {'Formula':24s}  {hdr_row}")
    print("-" * 72)

    for mi in all_mol_idx:
        # Find formula and atom count from whichever config has data for this mol
        formula = "?"
        atoms   = "?"
        for s in suffixes:
            row = data[s].get(mi, {})
            if row.get("formula"):
                formula = row["formula"]
                atoms   = row.get("actual_atoms", "?")
                break

        times = []
        for s in suffixes:
            row  = data[s].get(mi, {})
            if not row:
                times.append("—")
            elif row.get("status", "") != "ok":
                times.append("FAILED")
            else:
                times.append(row.get("wall_min", "—"))

        cols = "  ".join(fmt_min(t) for t in times)
        print(f"{atoms:>6s}  {formula:24s}  {cols}")

print()
print("Speedup vs 1-node = (1N time) / (N-node time).  — = pending/timed out.")