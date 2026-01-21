#!/usr/bin/env python
"""Phase 3 learnability check: train on backward, eval on backward."""

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


def load_summary(path: Path) -> Dict[str, float]:
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

    out_root = Path("runs/phase3_learnability")
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for variant in variants:
            alpha, beta = lock["alpha_beta"][variant]
            run_dir = out_root / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            ckpt = run_dir / "checkpoint_final.pt"
            if not ckpt.exists():
                run(
                    [
                        sys.executable,
                        "scripts/train.py",
                        "--task",
                        "backward",
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

            eval_dir = run_dir / "eval_backward"
            eval_dir.mkdir(parents=True, exist_ok=True)
            summary_path = eval_dir / "summary.json"
            if not summary_path.exists():
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
                        str(eval_dir),
                        "--phase3",
                    ]
                )

            summary = load_summary(summary_path)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "train_task": "backward",
                    "eval_task": "backward",
                    "accuracy": summary["accuracy"],
                    "checkpoint": str(ckpt),
                }
            )

    master_path = out_root / "phase3_learnability_master.csv"
    with master_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "seed", "train_task", "eval_task", "accuracy", "checkpoint"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def pooled_acc(name: str) -> float:
        vals = [r["accuracy"] for r in rows if r["variant"] == name]
        return float(sum(vals) / len(vals)) if vals else 0.0

    report_lines = [
        "# Phase 3 Learnability Report",
        "",
        f"pooled_backward_mechanism={pooled_acc('mechanism'):.4f}",
        f"pooled_backward_no_injection={pooled_acc('no_injection'):.4f}",
        "",
        "Per-seed backward accuracies:",
    ]
    for seed in seeds:
        mech = [r["accuracy"] for r in rows if r["variant"] == "mechanism" and r["seed"] == seed]
        base = [r["accuracy"] for r in rows if r["variant"] == "no_injection" and r["seed"] == seed]
        if mech and base:
            report_lines.append(f"- seed {seed}: mech={mech[0]:.4f} baseline={base[0]:.4f}")

    report_lines.extend(
        [
            "",
            "Conclusions:",
            "- Backward dataset learnability under direct supervision is summarized above.",
            "- Baseline >= 0.60 pooled indicates the backward task is learnable under supervision.",
            "- If baseline remains near chance, the generator or labels may be too ambiguous.",
            "- Mechanism vs baseline under supervision is exploratory and should be labeled as such.",
            "- Use these results to decide whether Phase 3b is meaningful.",
        ]
    )

    report_path = out_root / "phase3_learnability_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {master_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
