#!/usr/bin/env python
"""Run Phase 3 multitask (forward+backward) training and aggregate results."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def load_accuracy(path: Path) -> float:
    return float(json.loads(path.read_text())["accuracy"])


def main() -> int:
    lock = json.loads(Path("phase2_lock.json").read_text())
    seeds = lock["seeds"]
    train_size = lock["train"]["forward"]["size"]
    eval_size = lock["eval"]["forward"]["size"]
    steps = lock["train"]["forward"]["steps"]
    batch_size = lock["batch_size"]
    lr = lock["optimizer"]["lr"]
    weight_decay = lock["optimizer"]["weight_decay"]
    variants = ["mechanism", "no_injection"]

    out_root = Path("runs/phase3_multitask")
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for variant in variants:
            alpha, beta = lock["alpha_beta"][variant]
            run_dir = out_root / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            eval_fwd = run_dir / "eval_forward.json"
            eval_bwd = run_dir / "eval_backward.json"
            if not eval_fwd.exists() or not eval_bwd.exists():
                run(
                    [
                        sys.executable,
                        "scripts/phase3_multitask.py",
                        "--variant",
                        variant,
                        "--seed",
                        str(seed),
                        "--train_size",
                        str(train_size),
                        "--eval_size",
                        str(eval_size),
                        "--steps",
                        str(steps),
                        "--batch_size",
                        str(batch_size),
                        "--lr",
                        str(lr),
                        "--weight_decay",
                        str(weight_decay),
                        "--alpha",
                        str(alpha),
                        "--beta",
                        str(beta),
                        "--out_dir",
                        str(run_dir),
                    ]
                )

            acc_fwd = load_accuracy(eval_fwd)
            acc_bwd = load_accuracy(eval_bwd)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "train_task": "multitask",
                    "eval_task": "forward",
                    "accuracy": acc_fwd,
                    "run_dir": str(run_dir),
                }
            )
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "train_task": "multitask",
                    "eval_task": "backward",
                    "accuracy": acc_bwd,
                    "run_dir": str(run_dir),
                }
            )

    master_path = out_root / "phase3_multitask_master.csv"
    with master_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "seed", "train_task", "eval_task", "accuracy", "run_dir"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def pooled(variant: str, task: str) -> float:
        vals = [r["accuracy"] for r in rows if r["variant"] == variant and r["eval_task"] == task]
        return float(sum(vals) / len(vals)) if vals else 0.0

    report_lines = [
        "# Phase 3 Multitask Report",
        "",
        f"pooled_forward_mechanism={pooled('mechanism', 'forward'):.4f}",
        f"pooled_forward_no_injection={pooled('no_injection', 'forward'):.4f}",
        f"pooled_backward_mechanism={pooled('mechanism', 'backward'):.4f}",
        f"pooled_backward_no_injection={pooled('no_injection', 'backward'):.4f}",
        "",
        "Per-seed backward accuracies:",
    ]
    for seed in seeds:
        mech = [r["accuracy"] for r in rows if r["variant"] == "mechanism" and r["eval_task"] == "backward" and r["seed"] == seed]
        base = [r["accuracy"] for r in rows if r["variant"] == "no_injection" and r["eval_task"] == "backward" and r["seed"] == seed]
        if mech and base:
            report_lines.append(f"- seed {seed}: mech={mech[0]:.4f} baseline={base[0]:.4f}")

    report_path = out_root / "phase3_multitask_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {master_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
