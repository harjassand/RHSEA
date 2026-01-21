#!/usr/bin/env python
"""Phase 3b reverse-direction reciprocity: train backward, eval forward."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * ((phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_preds(path: Path) -> Dict[str, int]:
    preds: Dict[str, int] = {}
    labels: Dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["instance_id"]] = int(rec["pred"])
            labels[rec["instance_id"]] = int(rec["label"])
    return preds, labels


def bootstrap_delta(acc_a: List[int], acc_b: List[int], seed: int, n_boot: int = 10000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(acc_a)
    diffs = np.empty(n_boot, dtype=np.float64)
    arr_a = np.asarray(acc_a, dtype=np.float64)
    arr_b = np.asarray(acc_b, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(arr_a[idx] - arr_b[idx]))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def main() -> int:
    lock = json.loads(Path("phase2_lock.json").read_text())
    seeds = lock["seeds"]
    train_size = lock["train"]["forward"]["size"]
    train_steps = lock["train"]["forward"]["steps"]
    eval_size = lock["eval"]["forward"]["size"]
    batch_size = lock["batch_size"]
    lr = lock["optimizer"]["lr"]
    weight_decay = lock["optimizer"]["weight_decay"]

    variants = ["mechanism", "no_injection", "symmetric_control_v2_normmatched"]
    out_root = Path("runs/phase3b_reverse")
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    sel_rows: List[Dict[str, object]] = []
    for seed in seeds:
        ckpts: Dict[str, Path] = {}
        for variant in variants:
            alpha, beta = lock["alpha_beta"][variant]
            run_dir = out_root / "train_backward" / variant / f"seed_{seed}"
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
            ckpts[variant] = ckpt

            eval_dir = run_dir / "eval_forward"
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
                        "forward",
                        "--eval_size",
                        str(eval_size),
                        "--batch_size",
                        str(batch_size),
                        "--out_dir",
                        str(eval_dir),
                        "--phase3",
                    ]
                )
            summary = json.loads(summary_path.read_text())
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "train_task": "backward",
                    "eval_task": "forward",
                    "accuracy": summary["accuracy"],
                    "checkpoint": str(ckpt),
                    "eval_dir": str(eval_dir),
                }
            )

        # SelLocGap using paired mech vs sym control (v2).
        sel_dir = out_root / "sel_locgap" / f"seed_{seed}"
        sel_dir.mkdir(parents=True, exist_ok=True)
        sel_summary = sel_dir / "summary.json"
        if not sel_summary.exists():
            run(
                [
                    sys.executable,
                    "scripts/eval_forward_paired.py",
                    "--mech_ckpt",
                    str(ckpts["mechanism"]),
                    "--sym_ckpt",
                    str(ckpts["symmetric_control_v2_normmatched"]),
                    "--eval_size",
                    str(eval_size),
                    "--batch_size",
                    str(batch_size),
                    "--k_tok",
                    str(lock["k_tok"]),
                    "--bootstrap",
                    "10000",
                    "--out_dir",
                    str(sel_dir),
                    "--phase3",
                ]
            )
        sel = json.loads(sel_summary.read_text())
        sel_rows.append(
            {
                "seed": seed,
                "SelLocGap_mean": sel["SelLocGap"]["mean"],
                "SelLocGap_ci_low": sel["SelLocGap"]["ci_low"],
                "SelLocGap_ci_high": sel["SelLocGap"]["ci_high"],
            }
        )

    master_path = out_root / "phase3b_reverse_master.csv"
    with master_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "seed", "train_task", "eval_task", "accuracy", "checkpoint", "eval_dir"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Accuracy CIs + delta (mech - baseline).
    delta_rows: List[str] = []
    pooled_acc: Dict[str, Tuple[float, float, float]] = {}
    for variant in ["mechanism", "no_injection"]:
        acc_vals = [r for r in rows if r["variant"] == variant]
        all_correct = 0
        all_total = 0
        for r in acc_vals:
            eval_dir = Path(r["eval_dir"])
            preds, labels = load_preds(eval_dir / "eval_instances.jsonl.gz")
            correct = sum(1 for k, v in preds.items() if labels[k] == v)
            total = len(preds)
            all_correct += correct
            all_total += total
        acc = all_correct / all_total if all_total else 0.0
        ci_low, ci_high = wilson_ci(all_correct, all_total)
        pooled_acc[variant] = (acc, ci_low, ci_high)

    # Delta CI via bootstrap on paired instance correctness.
    delta_seed_vals: List[str] = []
    pooled_diff_samples: List[int] = []
    pooled_base_samples: List[int] = []
    pooled_mech_samples: List[int] = []
    for seed in seeds:
        mech_eval = out_root / "train_backward" / "mechanism" / f"seed_{seed}" / "eval_forward"
        base_eval = out_root / "train_backward" / "no_injection" / f"seed_{seed}" / "eval_forward"
        mech_preds, mech_labels = load_preds(mech_eval / "eval_instances.jsonl.gz")
        base_preds, base_labels = load_preds(base_eval / "eval_instances.jsonl.gz")
        common = sorted(set(mech_preds.keys()) & set(base_preds.keys()))
        mech_corr = [1 if mech_preds[k] == mech_labels[k] else 0 for k in common]
        base_corr = [1 if base_preds[k] == base_labels[k] else 0 for k in common]
        delta_mean = float(np.mean(np.asarray(mech_corr) - np.asarray(base_corr)))
        ci_low, ci_high = bootstrap_delta(mech_corr, base_corr, seed=seed)
        delta_seed_vals.append(f"- seed {seed}: delta={delta_mean:.4f} CI=[{ci_low:.4f},{ci_high:.4f}]")
        pooled_mech_samples.extend(mech_corr)
        pooled_base_samples.extend(base_corr)
    pooled_delta = float(np.mean(np.asarray(pooled_mech_samples) - np.asarray(pooled_base_samples)))
    pooled_ci_low, pooled_ci_high = bootstrap_delta(pooled_mech_samples, pooled_base_samples, seed=0)

    reverse_gap_present = pooled_delta <= -0.05 and pooled_ci_high < 0.0

    report_lines = [
        "# Phase 3b Reverse Reciprocity Report",
        "",
        f"pooled_forward_mechanism={pooled_acc['mechanism'][0]:.4f} CI=[{pooled_acc['mechanism'][1]:.4f},{pooled_acc['mechanism'][2]:.4f}]",
        f"pooled_forward_no_injection={pooled_acc['no_injection'][0]:.4f} CI=[{pooled_acc['no_injection'][1]:.4f},{pooled_acc['no_injection'][2]:.4f}]",
        f"pooled_delta(mech - baseline)={pooled_delta:.4f} CI=[{pooled_ci_low:.4f},{pooled_ci_high:.4f}]",
        f"reverse_gap_present={str(reverse_gap_present)}",
        "",
        "Per-seed delta (mech - baseline):",
    ]
    report_lines.extend(delta_seed_vals)
    report_lines.append("")
    report_lines.append("SelLocGap (mech - sym v2) per seed:")
    for row in sel_rows:
        report_lines.append(
            f"- seed {row['seed']}: mean={row['SelLocGap_mean']:.4f} CI=[{row['SelLocGap_ci_low']:.4f},{row['SelLocGap_ci_high']:.4f}]"
        )

    report_path = out_root / "phase3b_reverse_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {master_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
