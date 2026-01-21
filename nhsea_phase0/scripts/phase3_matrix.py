#!/usr/bin/env python
"""Phase 3 reciprocity test: forward-trained models evaluated on backward task."""

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


def _load_summary(path: Path) -> Dict[str, float]:
    return json.loads(path.read_text())


def main() -> int:
    lock = json.loads(Path("phase2_lock.json").read_text())
    seeds = lock["seeds"]
    train_size = lock["train"]["forward"]["size"]
    train_steps = lock["train"]["forward"]["steps"]
    eval_size = lock["eval"]["forward"]["size"]
    batch_size = lock["batch_size"]
    lr = lock["optimizer"]["lr"]
    weight_decay = lock["optimizer"]["weight_decay"]

    variants = ["mechanism", "no_injection"]
    out_root = Path("runs/phase3")
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for variant in variants:
            alpha, beta = lock["alpha_beta"][variant]
            run_dir = out_root / "forward_train" / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            ckpt = run_dir / "checkpoint_final.pt"
            if not ckpt.exists():
                run(
                    [
                        sys.executable,
                        "scripts/train.py",
                        "--task",
                        "forward",
                        "--variant",
                        variant,
                        "--alpha",
                        str(alpha),
                        "--beta",
                        str(beta),
                        "--seed",
                        str(seed),
                        "--train_size",
                        str(train_size),
                        "--steps",
                        str(train_steps),
                        "--batch_size",
                        str(batch_size),
                        "--lr",
                        str(lr),
                        "--weight_decay",
                        str(weight_decay),
                        "--phase3",
                        "--run_dir",
                        str(run_dir),
                        "--resume_if_exists",
                    ]
                )

            eval_forward_dir = run_dir / "eval_forward"
            eval_backward_dir = run_dir / "eval_backward"
            eval_forward_dir.mkdir(parents=True, exist_ok=True)
            eval_backward_dir.mkdir(parents=True, exist_ok=True)

            forward_summary = eval_forward_dir / "summary.json"
            if not forward_summary.exists():
                run(
                    [
                        sys.executable,
                        "scripts/eval_accuracy.py",
                        "--checkpoint",
                        str(ckpt),
                        "--task",
                        "forward",
                        "--eval_size",
                        str(eval_size),
                        "--batch_size",
                        str(batch_size),
                        "--out_dir",
                        str(eval_forward_dir),
                        "--phase3",
                    ]
                )

            backward_summary = eval_backward_dir / "summary.json"
            if not backward_summary.exists():
                run(
                    [
                        sys.executable,
                        "scripts/eval_accuracy.py",
                        "--checkpoint",
                        str(ckpt),
                        "--task",
                        "backward",
                        "--eval_size",
                        str(eval_size),
                        "--batch_size",
                        str(batch_size),
                        "--out_dir",
                        str(eval_backward_dir),
                        "--phase3",
                    ]
                )

            f_sum = _load_summary(forward_summary)
            b_sum = _load_summary(backward_summary)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "train_task": "forward",
                    "eval_task": "forward",
                    "accuracy": f_sum["accuracy"],
                    "checkpoint": str(ckpt),
                    "eval_dir": str(eval_forward_dir),
                }
            )
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "train_task": "forward",
                    "eval_task": "backward",
                    "accuracy": b_sum["accuracy"],
                    "checkpoint": str(ckpt),
                    "eval_dir": str(eval_backward_dir),
                }
            )

    master_path = out_root / "phase3_master.csv"
    with master_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "seed", "train_task", "eval_task", "accuracy", "checkpoint", "eval_dir"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Aggregate pooled backward accuracy and threshold check.
    mech_backward = [r["accuracy"] for r in rows if r["variant"] == "mechanism" and r["eval_task"] == "backward"]
    base_backward = [r["accuracy"] for r in rows if r["variant"] == "no_injection" and r["eval_task"] == "backward"]
    mech_forward = [r["accuracy"] for r in rows if r["variant"] == "mechanism" and r["eval_task"] == "forward"]
    base_forward = [r["accuracy"] for r in rows if r["variant"] == "no_injection" and r["eval_task"] == "forward"]

    mean_mech_backward = float(sum(mech_backward) / len(mech_backward)) if mech_backward else 0.0
    mean_base_backward = float(sum(base_backward) / len(base_backward)) if base_backward else 0.0
    mean_mech_forward = float(sum(mech_forward) / len(mech_forward)) if mech_forward else 0.0
    mean_base_forward = float(sum(base_forward) / len(base_forward)) if base_forward else 0.0

    delta_backward = mean_mech_backward - mean_base_backward
    threshold = -0.10
    prediction_met = delta_backward <= threshold
    falsified = not prediction_met

    report_lines = [
        "# Phase 3 Reciprocity Report",
        "",
        f"pooled_forward_mechanism={mean_mech_forward:.4f}",
        f"pooled_forward_no_injection={mean_base_forward:.4f}",
        f"pooled_backward_mechanism={mean_mech_backward:.4f}",
        f"pooled_backward_no_injection={mean_base_backward:.4f}",
        f"delta_backward(mech - baseline)={delta_backward:.4f}",
        f"falsification_threshold={threshold:.2f}",
        f"prediction_met={str(prediction_met)}",
        f"falsified={str(falsified)}",
        "",
        "Per-seed backward accuracies:",
    ]
    for seed in seeds:
        mech = [r["accuracy"] for r in rows if r["variant"] == "mechanism" and r["eval_task"] == "backward" and r["seed"] == seed]
        base = [r["accuracy"] for r in rows if r["variant"] == "no_injection" and r["eval_task"] == "backward" and r["seed"] == seed]
        if mech and base:
            report_lines.append(f"- seed {seed}: mech={mech[0]:.4f} baseline={base[0]:.4f} delta={mech[0]-base[0]:.4f}")

    report_path = out_root / "phase3_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {master_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
