#!/usr/bin/env python
"""Aggregate Claim B cycle metrics across variants and seeds."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


def _bootstrap_ci(values: np.ndarray, seed: int = 0, n_boot: int = 10000) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(values[idx]))
    return {"low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975))}


def _summary(values: List[float], seed: int = 0, n_boot: int = 10000) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    ci = _bootstrap_ci(arr, seed=seed, n_boot=n_boot)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci_low": ci["low"],
        "ci_high": ci["high"],
    }


def _diff_mean_ci(a_vals: List[float], b_vals: List[float], seed: int = 0, n_boot: int = 10000) -> Dict[str, float]:
    a = np.asarray(a_vals, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        a_idx = rng.integers(0, len(a), size=len(a))
        b_idx = rng.integers(0, len(b), size=len(b))
        boots[i] = float(np.mean(a[a_idx]) - np.mean(b[b_idx]))
    return {
        "mean": float(np.mean(a) - np.mean(b)),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


def _parse_run_id(run_id: str) -> Dict[str, str] | None:
    match = re.match(
        r"^phase2_v2_cycle_(?P<variant>mechanism|no_injection|no_drift|symmetric_control_v2_normmatched)_seed(?P<seed>\d+)$",
        run_id,
    )
    if not match:
        return None
    return match.groupdict()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs/phase2_claimB")
    ap.add_argument("--out_md", type=str, default="")
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    root = Path(args.root)
    out_md = Path(args.out_md) if args.out_md else root / "phase2_claimB_report.md"
    out_csv = Path(args.out_csv) if args.out_csv else root / "phase2_claimB_master.csv"

    # Collect per-instance metrics.
    data: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}
    accuracy: Dict[str, Dict[str, float]] = {}

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        meta = _parse_run_id(run_dir.name)
        if meta is None:
            continue
        variant = meta["variant"]
        seed = meta["seed"]
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            accuracy.setdefault(variant, {})[seed] = float(summary.get("accuracy", 0.0))

        inst_path = run_dir / "eval_instances.jsonl.gz"
        if not inst_path.exists():
            continue

        data.setdefault(variant, {}).setdefault(seed, {}).setdefault("A", {}).setdefault("PR", {})
        data.setdefault(variant, {}).setdefault(seed, {}).setdefault("B", {}).setdefault("PR", {})
        for pipe in ("A", "B"):
            for ell in (2, 3, 4):
                data[variant][seed].setdefault(pipe, {}).setdefault(f"DeltaCyc{ell}", {})

        with gzip.open(inst_path, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                regime = str(rec.get("regime"))
                if regime is None:
                    continue
                data[variant][seed]["A"]["PR"].setdefault(regime, []).append(rec["pr_a"])
                data[variant][seed]["B"]["PR"].setdefault(regime, []).append(rec["pr_b"])
                for ell in (2, 3, 4):
                    key = str(ell)
                    data[variant][seed]["A"][f"DeltaCyc{ell}"].setdefault(regime, []).append(rec["delta_cyc_a"][key])
                    data[variant][seed]["B"][f"DeltaCyc{ell}"].setdefault(regime, []).append(rec["delta_cyc_b"][key])

    rows: List[Dict[str, str]] = []
    report_lines: List[str] = ["# Phase 2 Claim B Report", ""]

    for variant, seeds in data.items():
        report_lines.append(f"## Variant: {variant}")
        for seed, pipes in seeds.items():
            acc = accuracy.get(variant, {}).get(seed, 0.0)
            report_lines.append(f"- seed {seed}: accuracy={acc:.4f}")
            for pipe in ("A", "B"):
                for regime in ("0", "1", "2", "3"):
                    pr_stats = _summary(pipes[pipe]["PR"].get(regime, []), seed=0, n_boot=args.bootstrap)
                    rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "pipeline": pipe,
                            "regime": regime,
                            "metric": "PR",
                            "ell": "",
                            "mean": pr_stats["mean"],
                            "median": pr_stats["median"],
                            "ci_low": pr_stats["ci_low"],
                            "ci_high": pr_stats["ci_high"],
                            "accuracy": acc,
                        }
                    )
                    for ell in (2, 3, 4):
                        stats = _summary(pipes[pipe][f"DeltaCyc{ell}"].get(regime, []), seed=0, n_boot=args.bootstrap)
                        rows.append(
                            {
                                "variant": variant,
                                "seed": seed,
                                "pipeline": pipe,
                                "regime": regime,
                                "metric": "DeltaCyc",
                                "ell": str(ell),
                                "mean": stats["mean"],
                                "median": stats["median"],
                                "ci_low": stats["ci_low"],
                                "ci_high": stats["ci_high"],
                                "accuracy": acc,
                            }
                        )

        # pooled across seeds
        report_lines.append("- pooled across seeds:")
        for pipe in ("A", "B"):
            pr_cyc: List[float] = []
            pr_dag: List[float] = []
            delta_cyc: Dict[int, List[float]] = {2: [], 3: [], 4: []}
            delta_dag: Dict[int, List[float]] = {2: [], 3: [], 4: []}
            for seed, pipes in seeds.items():
                pr_dag.extend(pipes[pipe]["PR"].get("0", []))
                for r in ("1", "2", "3"):
                    pr_cyc.extend(pipes[pipe]["PR"].get(r, []))
                for ell in (2, 3, 4):
                    delta_dag[ell].extend(pipes[pipe][f"DeltaCyc{ell}"].get("0", []))
                    for r in ("1", "2", "3"):
                        delta_cyc[ell].extend(pipes[pipe][f"DeltaCyc{ell}"].get(r, []))
            pr_diff = _diff_mean_ci(pr_cyc, pr_dag, seed=0, n_boot=args.bootstrap)
            report_lines.append(
                f"  PR_{pipe}(cyc-DAG) mean={pr_diff['mean']:.6f} "
                f"CI=[{pr_diff['ci_low']:.6f},{pr_diff['ci_high']:.6f}]"
            )
            for ell in (2, 3, 4):
                diff = _diff_mean_ci(delta_cyc[ell], delta_dag[ell], seed=0, n_boot=args.bootstrap)
                report_lines.append(
                    f"  ΔCyc{ell}_{pipe}(cyc-DAG) mean={diff['mean']:.6f} "
                    f"CI=[{diff['ci_low']:.6f},{diff['ci_high']:.6f}]"
                )

        report_lines.append("")

    # Write CSV
    if rows:
        keys = ["variant", "seed", "pipeline", "regime", "metric", "ell", "mean", "median", "ci_low", "ci_high", "accuracy"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        out_csv.write_text("")

    out_md.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
