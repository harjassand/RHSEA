#!/usr/bin/env python
"""Phase 3 reciprocity test: zero-shot + fine-tune curve on backward task."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    lock = json.loads(Path("phase2_lock.json").read_text())
    seeds = lock["seeds"]
    eval_size = lock["eval"]["forward"]["size"]
    batch_size = lock["batch_size"]
    train_steps = lock["train"]["forward"]["steps"]
    lr = lock["optimizer"]["lr"]
    weight_decay = lock["optimizer"]["weight_decay"]
    sizes = [100, 1000, 10000, lock["train"]["forward"]["size"]]

    out_root = Path("runs/phase3/reciprocity")
    out_root.mkdir(parents=True, exist_ok=True)

    variants = ["mechanism", "no_injection"]
    rows = []

    # Protocol A: zero-shot
    for seed in seeds:
        for variant in variants:
            ckpt = Path("runs/phase2/forward") / variant / f"seed_{seed}" / "checkpoint.pt"
            out_dir = out_root / "zero_shot" / variant / f"seed_{seed}"
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
                    str(out_dir),
                ]
            )
            summary = json.loads((out_dir / "summary.json").read_text())
            rows.append(
                {
                    "protocol": "zero_shot",
                    "variant": variant,
                    "seed": seed,
                    "train_size": 0,
                    "accuracy": summary["accuracy"],
                }
            )

    # Protocol B: fine-tune curve
    for seed in seeds:
        for variant in variants:
            base_ckpt = Path("runs/phase2/forward") / variant / f"seed_{seed}" / "checkpoint.pt"
            for n in sizes:
                run_dir = out_root / "finetune" / variant / f"seed_{seed}" / f"n_{n}"
                run_dir.mkdir(parents=True, exist_ok=True)
                run(
                    [
                        sys.executable,
                        "scripts/train.py",
                        "--task",
                        "backward",
                        "--variant",
                        variant,
                        "--alpha",
                        str(lock["alpha_beta"][variant][0]),
                        "--beta",
                        str(lock["alpha_beta"][variant][1]),
                        "--seed",
                        str(seed),
                        "--train_size",
                        str(n),
                        "--steps",
                        str(train_steps),
                        "--batch_size",
                        str(batch_size),
                        "--lr",
                        str(lr),
                        "--weight_decay",
                        str(weight_decay),
                        "--init_checkpoint",
                        str(base_ckpt),
                        "--run_dir",
                        str(run_dir),
                    ]
                )
                run(
                    [
                        sys.executable,
                        "scripts/eval_accuracy.py",
                        "--checkpoint",
                        str(run_dir / "checkpoint.pt"),
                        "--task",
                        "backward",
                        "--eval_size",
                        str(eval_size),
                        "--batch_size",
                        str(batch_size),
                        "--out_dir",
                        str(run_dir),
                    ]
                )
                summary = json.loads((run_dir / "summary.json").read_text())
                rows.append(
                    {
                        "protocol": "finetune",
                        "variant": variant,
                        "seed": seed,
                        "train_size": n,
                        "accuracy": summary["accuracy"],
                    }
                )

    # Write summary CSV
    csv_path = out_root / "reciprocity_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["protocol", "variant", "seed", "train_size", "accuracy"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    report_path = out_root / "reciprocity_report.md"
    report_lines = ["# Reciprocity Test Summary", "", f"eval_size={eval_size}", ""]
    for r in rows:
        report_lines.append(
            f"- {r['protocol']} variant={r['variant']} seed={r['seed']} n={r['train_size']} acc={r['accuracy']:.4f}"
        )
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
