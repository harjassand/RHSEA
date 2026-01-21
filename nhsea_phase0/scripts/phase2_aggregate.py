#!/usr/bin/env python
"""Aggregate Phase 2 results into CSV + report."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


ROOT = Path("runs/phase2")


def _bootstrap_ci(values: np.ndarray, seed: int = 0, n_boot: int = 10000) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(values[idx]))
    return {"low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975))}


def _mean_ci(values: List[float], seed: int = 0, n_boot: int = 10000) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    ci = _bootstrap_ci(arr, seed=seed, n_boot=n_boot)
    return {"mean": float(np.mean(arr)), "ci_low": ci["low"], "ci_high": ci["high"]}


def _diff_mean_ci(a_vals: List[float], b_vals: List[float], seed: int = 0, n_boot: int = 10000) -> Dict[str, float]:
    a = np.asarray(a_vals, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
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


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _iter_summary_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/summary.json"))


def _parse_run_path(path: Path, root: Path) -> Optional[Dict[str, str]]:
    # supports both:
    # runs/phase2/{task}/{variant}/seed_{s}/summary.json
    # runs/phase2_restart_v2/phase2_v2_{task}_{variant}_seed{s}/summary.json
    rel = path.relative_to(root)
    if len(rel.parts) >= 4:
        task = rel.parts[0]
        variant = rel.parts[1]
        seed = rel.parts[2].replace("seed_", "")
        return {"task": task, "variant": variant, "seed": seed}
    if len(rel.parts) >= 2:
        run_id = rel.parts[0]
        match = re.match(
            r"^phase2_v2_(?P<task>forward|cycle)_(?P<variant>mechanism|symmetric_control|symmetric_control_v2_normmatched|no_injection|no_drift)_seed(?P<seed>\d+)$",
            run_id,
        )
        if match:
            return match.groupdict()
    return None


def write_aggregate_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(ROOT))
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--master_out", type=str, default="")
    ap.add_argument("--report_out", type=str, default="")
    args = ap.parse_args()
    root = Path(args.root)
    rows: List[Dict[str, str]] = []

    run_entries: List[Dict[str, str]] = []
    for summary_path in _iter_summary_files(root):
        meta = _parse_run_path(summary_path, root)
        if meta is None:
            continue
        meta = dict(meta)
        meta["summary_path"] = str(summary_path)
        meta["run_dir"] = str(summary_path.parent)
        run_entries.append(meta)

    for entry in run_entries:
        summary_path = Path(entry["summary_path"])
        data = _read_json(summary_path)
        row = {
            "task": entry["task"],
            "variant": entry["variant"],
            "seed": entry["seed"],
            "accuracy": data.get("accuracy", ""),
            "alpha": data.get("alpha", data.get("alpha_mech", "")),
            "beta": data.get("beta", data.get("beta_mech", "")),
            "k_tok": data.get("k_tok", ""),
            "k_prop": data.get("k_prop", ""),
            "rho_median": data.get("rho_median", ""),
            "rho_flag": data.get("rho_flag", ""),
            "A_median": data.get("A_median", ""),
            "B_median": data.get("B_median", ""),
        }
        if "AdjLocGap" in data:
            row.update(
                {
                    "AdjLocGap_mean": data["AdjLocGap"]["mean"],
                    "AdjLocGap_median": data["AdjLocGap"]["median"],
                    "AdjLocGap_ci_low": data["AdjLocGap"]["ci_low"],
                    "AdjLocGap_ci_high": data["AdjLocGap"]["ci_high"],
                }
            )
        if "SelLocGap" in data:
            row.update(
                {
                    "SelLocGap_mean": data["SelLocGap"]["mean"],
                    "SelLocGap_median": data["SelLocGap"]["median"],
                    "SelLocGap_ci_low": data["SelLocGap"]["ci_low"],
                    "SelLocGap_ci_high": data["SelLocGap"]["ci_high"],
                    "sign_test_p": data.get("sign_test_p", ""),
                }
            )
        rows.append(row)

    out_csv = Path(args.out) if args.out else root / "phase2_aggregate.csv"
    write_aggregate_csv(rows, out_csv)
    out_json = Path(args.out_json) if args.out_json else root / "phase2_aggregate.json"
    out_json.write_text(json.dumps(rows, indent=2, sort_keys=True))

    master_rows: List[Dict[str, str]] = []
    for entry in run_entries:
        summary_path = Path(entry["summary_path"])
        data = _read_json(summary_path)
        base = {
            "task": entry["task"],
            "variant": entry["variant"],
            "seed": entry["seed"],
            "accuracy": data.get("accuracy", ""),
            "alpha": data.get("alpha", data.get("alpha_mech", "")),
            "beta": data.get("beta", data.get("beta_mech", "")),
            "k_tok": data.get("k_tok", ""),
            "k_prop": data.get("k_prop", ""),
            "rho_median": data.get("rho_median", ""),
            "rho_flag": data.get("rho_flag", ""),
            "A_median": data.get("A_median", ""),
            "B_median": data.get("B_median", ""),
        }
        if entry["task"] == "cycle":
            for pipe in ("A", "B"):
                row = dict(base)
                row["pipeline"] = pipe
                row["DeltaCyc"] = json.dumps(data.get(f"DeltaCyc_{pipe}", {}))
                row["PR"] = json.dumps(data.get(f"PR_{pipe}", {}))
                master_rows.append(row)
        else:
            row = dict(base)
            row["pipeline"] = "token"
            if "AdjLocGap" in data:
                row["AdjLocGap"] = json.dumps(data["AdjLocGap"])
            master_rows.append(row)
    master_out = Path(args.master_out) if args.master_out else root / "phase2_master.csv"
    write_aggregate_csv(master_rows, master_out)

    # Build report
    report_lines: List[str] = []
    report_lines.append("# Phase 2 Report")
    report_lines.append("")

    # Claim A pooled SelLocGap across seeds
    paired_dirs: List[Path] = []
    for base in (root / "forward/paired_mech_minus_sym", root / "paired_mech_minus_sym"):
        if base.exists():
            paired_dirs.extend(sorted(base.glob("seed_*")))
    pooled_sel: List[float] = []
    seed_lines: List[str] = []
    valid_seeds = 0
    for d in paired_dirs:
        summary = _read_json(d / "summary.json")
        seed = d.name.replace("seed_", "")
        flag = summary.get("rho_flag", True)
        rho = summary.get("rho_median", float("nan"))
        seed_lines.append(
            f"- seed {seed}: SelLocGap mean={summary['SelLocGap']['mean']:.6f} "
            f"CI=[{summary['SelLocGap']['ci_low']:.6f},{summary['SelLocGap']['ci_high']:.6f}] "
            f"rho_median={rho:.6f} rho_flag={flag}"
        )
        if not flag:
            valid_seeds += 1
        inst_path = d / "eval_instances.jsonl.gz"
        with gzip.open(inst_path, "rt", encoding="utf-8") as f:
            for line in f:
                pooled_sel.append(json.loads(line)["sel_loc_gap"])

    if pooled_sel:
        pooled = _mean_ci(pooled_sel, seed=0, n_boot=10000)
        report_lines.append("## Claim A (SelLocGap)")
        report_lines.extend(seed_lines)
        report_lines.append(
            f"- pooled SelLocGap mean={pooled['mean']:.6f} CI=[{pooled['ci_low']:.6f},{pooled['ci_high']:.6f}]"
        )
        if valid_seeds == 0:
            report_lines.append("- status: NOT TESTABLE (all sym-controls flagged)")
        else:
            report_lines.append("- status: see CI and rho_flag per seed")

    # Claim B (mechanism cycle task)
    report_lines.append("")
    report_lines.append("## Claim B (Cycle diagnostics, mechanism)")
    mech_runs = [e for e in run_entries if e["task"] == "cycle" and e["variant"] == "mechanism"]
    for entry in mech_runs:
        seed = entry["seed"]
        inst_path = Path(entry["run_dir"]) / "eval_instances.jsonl.gz"
        pr_a_dag, pr_a_cyc = [], []
        pr_b_dag, pr_b_cyc = [], []
        delta_a = {2: {"dag": [], "cyc": []}, 3: {"dag": [], "cyc": []}, 4: {"dag": [], "cyc": []}}
        delta_b = {2: {"dag": [], "cyc": []}, 3: {"dag": [], "cyc": []}, 4: {"dag": [], "cyc": []}}
        with gzip.open(inst_path, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                regime = rec.get("regime")
                if regime is None:
                    continue
                is_cyc = int(regime) in (1, 2, 3)
                if is_cyc:
                    pr_a_cyc.append(rec["pr_a"])
                    pr_b_cyc.append(rec["pr_b"])
                else:
                    pr_a_dag.append(rec["pr_a"])
                    pr_b_dag.append(rec["pr_b"])
                for ell in (2, 3, 4):
                    key = str(ell)
                    if is_cyc:
                        delta_a[ell]["cyc"].append(rec["delta_cyc_a"][key])
                        delta_b[ell]["cyc"].append(rec["delta_cyc_b"][key])
                    else:
                        delta_a[ell]["dag"].append(rec["delta_cyc_a"][key])
                        delta_b[ell]["dag"].append(rec["delta_cyc_b"][key])

        if pr_a_cyc and pr_a_dag:
            pr_a_stats = _diff_mean_ci(pr_a_cyc, pr_a_dag, seed=0, n_boot=10000)
            pr_b_stats = _diff_mean_ci(pr_b_cyc, pr_b_dag, seed=0, n_boot=10000)
            report_lines.append(
                f"- seed {seed}: PR_A(cyc-DAG) mean={pr_a_stats['mean']:.6f} CI=[{pr_a_stats['ci_low']:.6f},{pr_a_stats['ci_high']:.6f}]"
            )
            report_lines.append(
                f"  PR_B(cyc-DAG) mean={pr_b_stats['mean']:.6f} CI=[{pr_b_stats['ci_low']:.6f},{pr_b_stats['ci_high']:.6f}]"
            )

        for ell in (2, 3, 4):
            if delta_a[ell]["cyc"] and delta_a[ell]["dag"]:
                stats = _diff_mean_ci(delta_a[ell]["cyc"], delta_a[ell]["dag"], seed=0, n_boot=10000)
                report_lines.append(
                    f"  ΔCyc{ell}_A(cyc-DAG) mean={stats['mean']:.6f} CI=[{stats['ci_low']:.6f},{stats['ci_high']:.6f}]"
                )

    report_path = Path(args.report_out) if args.report_out else root / "phase2_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
