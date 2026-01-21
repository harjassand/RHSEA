#!/usr/bin/env python
"""Aggregate Phase 3c few-shot adaptation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def bootstrap_ci(values: List[float], seed: int = 0, n_boot: int = 10000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(arr[idx]))
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--report", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.root)
    summaries = list(root.glob("**/adapt_summary.json"))
    rows: List[Dict[str, object]] = []
    for path in summaries:
        data = json.loads(path.read_text())
        rows.append(
            {
                "variant": data["variant"],
                "seed": data["seed"],
                "source_task": data["source_task"],
                "target_task": data["target_task"],
                "tier": data["tier"],
                "n_train": data["n_train"],
                "final_acc": data["final_acc"],
                "best_acc": data["best_acc"],
                "step_at_best": data["step_at_best"],
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "seed",
                "source_task",
                "target_task",
                "tier",
                "n_train",
                "final_acc",
                "best_acc",
                "step_at_best",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Pooled curves + delta curves.
    directions = sorted({(r["source_task"], r["target_task"]) for r in rows})
    n_trains = sorted({int(r["n_train"]) for r in rows})

    report_lines = ["# Phase 3c Few-Shot Report", ""]
    adaptation_penalty_present = False

    for source_task, target_task in directions:
        report_lines.append(f"## {source_task} → {target_task}")
        for n in n_trains:
            for variant in ["mechanism", "no_injection"]:
                vals = [
                    r["final_acc"]
                    for r in rows
                    if r["variant"] == variant
                    and r["source_task"] == source_task
                    and r["target_task"] == target_task
                    and int(r["n_train"]) == n
                ]
                if not vals:
                    continue
                mean_val = float(np.mean(vals))
                ci_low, ci_high = bootstrap_ci(vals, seed=0)
                report_lines.append(
                    f"- {variant} n={n}: mean={mean_val:.4f} CI=[{ci_low:.4f},{ci_high:.4f}]"
                )

            # Delta (mech - baseline)
            mech_vals = [
                r["final_acc"]
                for r in rows
                if r["variant"] == "mechanism"
                and r["source_task"] == source_task
                and r["target_task"] == target_task
                and int(r["n_train"]) == n
            ]
            base_vals = [
                r["final_acc"]
                for r in rows
                if r["variant"] == "no_injection"
                and r["source_task"] == source_task
                and r["target_task"] == target_task
                and int(r["n_train"]) == n
            ]
            if mech_vals and base_vals and len(mech_vals) == len(base_vals):
                deltas = [m - b for m, b in zip(mech_vals, base_vals)]
                delta_mean = float(np.mean(deltas))
                ci_low, ci_high = bootstrap_ci(deltas, seed=0)
                report_lines.append(
                    f"- delta n={n}: mean={delta_mean:.4f} CI=[{ci_low:.4f},{ci_high:.4f}]"
                )
                if delta_mean <= -0.05 and ci_high < 0.0:
                    adaptation_penalty_present = True

        report_lines.append("")

    report_lines.append(f"adaptation_penalty_present={str(adaptation_penalty_present)}")

    report_path = Path(args.report)
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {out_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
