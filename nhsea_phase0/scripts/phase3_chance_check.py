#!/usr/bin/env python
"""Compute Wilson 95% CI for Phase 3 backward accuracy."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Tuple


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * ((phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_counts(path: Path) -> Tuple[int, int]:
    correct = 0
    total = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            if rec["pred"] == rec["label"]:
                correct += 1
    return correct, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs/phase3/forward_train")
    ap.add_argument("--out", type=str, default="runs/phase3/phase3_chance_check.json")
    ap.add_argument("--variants", nargs="+", default=["mechanism", "no_injection"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--eval_dir", type=str, default="eval_backward")
    args = ap.parse_args()

    root = Path(args.root)
    report: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant in args.variants:
        report[variant] = {}
        pooled_correct = 0
        pooled_total = 0
        for seed in args.seeds:
            inst_path = root / variant / f"seed_{seed}" / args.eval_dir / "eval_instances.jsonl.gz"
            correct, total = load_counts(inst_path)
            pooled_correct += correct
            pooled_total += total
            acc = correct / total if total else 0.0
            ci_low, ci_high = wilson_ci(correct, total)
            report[variant][f"seed_{seed}"] = {
                "n": total,
                "accuracy": acc,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "includes_chance": bool(ci_low <= 0.5 <= ci_high),
            }
        pooled_acc = pooled_correct / pooled_total if pooled_total else 0.0
        ci_low, ci_high = wilson_ci(pooled_correct, pooled_total)
        report[variant]["pooled"] = {
            "n": pooled_total,
            "accuracy": pooled_acc,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "includes_chance": bool(ci_low <= 0.5 <= ci_high),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
