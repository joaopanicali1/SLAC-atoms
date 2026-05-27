#!/usr/bin/env python3
"""Monitor beamstop Optuna SLURM jobs and surface failures."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


PROJECT_DIR = Path(os.environ.get(
    "SLAC_ATOMS_PROJECT_DIR",
    Path(__file__).resolve().parents[1],
)).resolve()
DB_PATH = PROJECT_DIR / "checkpoints/optuna/beamstop.db"
LOG_DIR = PROJECT_DIR / "training/logs"


def run_cmd(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return result.stderr.strip()
    return result.stdout.strip()


def squeue_status(job_name: str) -> tuple[str, list[str]]:
    user = os.environ.get("USER", "")
    out = run_cmd(["squeue", "-u", user, "-n", job_name, "-h", "-o", "%i %t %M %R"])
    if not out:
        return "No matching SLURM jobs in queue.", []
    if out.startswith("squeue:"):
        return out, []

    rows = out.splitlines()
    job_ids = [row.split()[0] for row in rows if row.split()]
    counts: dict[str, int] = {}
    for row in rows:
        parts = row.split()
        if len(parts) >= 2:
            counts[parts[1]] = counts.get(parts[1], 0) + 1
    summary = ", ".join(f"{state}={count}" for state, count in sorted(counts.items()))
    return f"SLURM: {summary} ({len(rows)} jobs)", job_ids


def optuna_status(study_name: str, db_path: Path) -> str:
    import optuna

    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
    )
    counts: dict[str, int] = {}
    for trial in study.trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1

    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    complete.sort(key=lambda t: t.value)
    best = ""
    if complete:
        best = f", best={complete[0].value:.6e} (trial #{complete[0].number})"
    count_s = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no trials"
    return f"Optuna: {count_s}{best}"


def current_stderr_alerts(log_dir: Path, job_ids: list[str]) -> list[str]:
    alerts = []
    for job_id in job_ids:
        err = log_dir / f"optuna_{job_id}.err"
        if err.exists() and err.stat().st_size > 0:
            tail = err.read_text(errors="replace").splitlines()[-3:]
            alerts.append(f"{err.name}: " + " | ".join(tail))
    return alerts


def check_once(args) -> bool:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    slurm_summary, job_ids = squeue_status(args.job_name)
    optuna_summary = optuna_status(args.study_name, args.db_path)
    alerts = current_stderr_alerts(args.log_dir, job_ids)

    print(f"\n[{timestamp}]")
    print(slurm_summary)
    print(optuna_summary)
    if alerts:
        print("ALERT: stderr output detected for current jobs:")
        for alert in alerts:
            print(f"  {alert}")
    return bool(alerts) or "FAIL=" in optuna_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="beamstop_v2")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--job-name", default="bs_optuna")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        failed = check_once(args)
        if args.once:
            raise SystemExit(1 if failed else 0)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
