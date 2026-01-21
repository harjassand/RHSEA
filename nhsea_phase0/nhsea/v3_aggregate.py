"""Aggregate NHSEA v3 baseline preflight runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * ((phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sum_counts(rows: List[dict]) -> Tuple[int, int]:
    correct = sum(int(r.get("correct", 0)) for r in rows)
    total = sum(int(r.get("total", 0)) for r in rows)
    return correct, total


def _bootstrap_ci(values: List[float], n_boot: int = 1000, seed: int = 0) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=np.float64)
    n = len(vals)
    means = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        means[i] = float(np.mean(sample))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--report", type=str, required=True)
    ap.add_argument("--json", type=str, required=True)
    args = ap.parse_args(argv)

    root = Path(args.root)
    eval_summaries = list(root.glob("**/eval_*_summary.json"))
    adapt_summaries = list(root.glob("**/adapt_summary.json"))

    rows: List[Dict[str, object]] = []
    for path in eval_summaries:
        data = _load_json(path)
        rows.append(
            {
                "mode": data.get("mode", ""),
                "train_task": data.get("train_task", ""),
                "eval_task": data.get("eval_task", ""),
                "source_task": data.get("train_task", ""),
                "target_task": data.get("eval_task", ""),
                "variant": data.get("variant", ""),
                "seed": data.get("seed", ""),
                "n_train": "",
                "acc": data.get("acc", 0.0),
                "ci_low": data.get("ci_low", 0.0),
                "ci_high": data.get("ci_high", 0.0),
                "correct": data.get("correct", 0),
                "total": data.get("total", 0),
                "path": str(path),
            }
        )

    for path in adapt_summaries:
        data = _load_json(path)
        rows.append(
            {
                "mode": "few_shot",
                "train_task": data.get("source_task", ""),
                "eval_task": data.get("target_task", ""),
                "source_task": data.get("source_task", ""),
                "target_task": data.get("target_task", ""),
                "variant": data.get("variant", ""),
                "seed": data.get("seed", ""),
                "n_train": data.get("n_train", ""),
                "acc": data.get("final_acc", 0.0),
                "ci_low": data.get("final_ci_low", 0.0),
                "ci_high": data.get("final_ci_high", 0.0),
                "correct": data.get("final_correct", 0),
                "total": data.get("final_total", 0),
                "path": str(path),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "train_task",
                "eval_task",
                "source_task",
                "target_task",
                "variant",
                "seed",
                "n_train",
                "acc",
                "ci_low",
                "ci_high",
                "correct",
                "total",
                "path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def _filter(mode: str, train_task: str, eval_task: str, n_train: int | None = None) -> List[dict]:
        out = [r for r in rows if r["mode"] == mode and r["train_task"] == train_task and r["eval_task"] == eval_task]
        if n_train is not None:
            out = [r for r in out if int(r.get("n_train") or 0) == n_train]
        return out

    def _acc_line(label: str, subset: List[dict]) -> Tuple[str, dict]:
        correct, total = _sum_counts(subset)
        acc = correct / total if total else 0.0
        ci_low, ci_high = wilson_ci(correct, total)
        return f"- {label}: acc={acc:.4f} CI=[{ci_low:.4f},{ci_high:.4f}] n={total}", {
            "acc": float(acc),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "correct": int(correct),
            "total": int(total),
        }

    leak_dir = root / "leak"
    leak_fwd = leak_dir / "leak_gate_v3_forward.json"
    leak_bwd = leak_dir / "leak_gate_v3_backward.json"
    leak_fwd_data = _load_json(leak_fwd) if leak_fwd.exists() else None
    leak_bwd_data = _load_json(leak_bwd) if leak_bwd.exists() else None

    # Topology diffs
    pair_files = list(root.glob("topology_pairs_seed*.json"))
    pair_rows: List[dict] = []
    for path in pair_files:
        data = _load_json(path)
        if isinstance(data, list):
            pair_rows.extend(data)

    pr_diffs = [float(r.get("pr_diff", 0.0)) for r in pair_rows if "pr_diff" in r]
    mass_diffs = [float(r.get("mass_diff", 0.0)) for r in pair_rows if "mass_diff" in r]

    pr_mean = float(np.mean(pr_diffs)) if pr_diffs else 0.0
    mass_mean = float(np.mean(mass_diffs)) if mass_diffs else 0.0
    pr_ci = _bootstrap_ci(pr_diffs) if pr_diffs else (0.0, 0.0)
    mass_ci = _bootstrap_ci(mass_diffs) if mass_diffs else (0.0, 0.0)

    report_lines = ["# V3 Preflight Report", ""]

    report_lines.append("## Leak gate")
    if leak_fwd_data:
        report_lines.append(f"- forward: AUROC={leak_fwd_data['auroc']:.4f} pass={leak_fwd_data['passed']}")
    else:
        report_lines.append("- forward: missing leak_gate_v3_forward.json")
    if leak_bwd_data:
        report_lines.append(f"- backward: AUROC={leak_bwd_data['auroc']:.4f} pass={leak_bwd_data['passed']}")
    else:
        report_lines.append("- backward: missing leak_gate_v3_backward.json")
    report_lines.append("")

    report_lines.append("## In-task accuracy")
    line, conc_metrics = _acc_line("conclusion", _filter("in_task", "conclusion", "conclusion"))
    report_lines.append(line)
    line, topo_metrics = _acc_line("topology", _filter("in_task", "topology", "topology"))
    report_lines.append(line)
    report_lines.append("")

    report_lines.append("## Zero-shot transfer")
    line, zf_metrics = _acc_line("conclusion→topology", _filter("zero_shot", "conclusion", "topology"))
    report_lines.append(line)
    line, zb_metrics = _acc_line("topology→conclusion", _filter("zero_shot", "topology", "conclusion"))
    report_lines.append(line)
    report_lines.append("")

    report_lines.append("## Few-shot transfer (head-only)")
    n_train_values = sorted({int(r.get("n_train") or 0) for r in rows if r["mode"] == "few_shot"})
    fewshot_metrics: Dict[str, dict] = {}
    for n_train in n_train_values:
        line, metrics = _acc_line(
            f"conclusion→topology n={n_train}",
            _filter("few_shot", "conclusion", "topology", n_train),
        )
        report_lines.append(line)
        fewshot_metrics[f"conclusion_to_topology_n{n_train}"] = metrics
        line, metrics = _acc_line(
            f"topology→conclusion n={n_train}",
            _filter("few_shot", "topology", "conclusion", n_train),
        )
        report_lines.append(line)
        fewshot_metrics[f"topology_to_conclusion_n{n_train}"] = metrics
    report_lines.append("")

    report_lines.append("## Topology sensitivity (paired OBC vs PBC)")
    report_lines.append(
        f"- PR diff (OBC-PBC): mean={pr_mean:.4f} CI=[{pr_ci[0]:.4f},{pr_ci[1]:.4f}] n_pairs={len(pr_diffs)}"
    )
    report_lines.append(
        f"- Mass diff (OBC-PBC): mean={mass_mean:.4f} CI=[{mass_ci[0]:.4f},{mass_ci[1]:.4f}] n_pairs={len(mass_diffs)}"
    )
    report_lines.append("")

    leak_pass = bool(leak_fwd_data and leak_bwd_data and leak_fwd_data.get("passed") and leak_bwd_data.get("passed"))
    in_task_pass = bool(conc_metrics["ci_low"] > 0.5 and topo_metrics["ci_low"] > 0.5)

    n_train_gate = 512 if 512 in n_train_values else (max(n_train_values) if n_train_values else None)
    transfer_pass = False
    if n_train_gate is not None:
        fwd = _filter("few_shot", "conclusion", "topology", n_train_gate)
        bwd = _filter("few_shot", "topology", "conclusion", n_train_gate)
        fwd_metrics = _acc_line("", fwd)[1]
        bwd_metrics = _acc_line("", bwd)[1]
        transfer_pass = bool(fwd_metrics["ci_low"] > 0.5 or bwd_metrics["ci_low"] > 0.5)
    pr_pass = pr_ci[1] < 0.0
    mass_pass = mass_ci[0] > 0.0
    topo_pass = bool(pr_pass or mass_pass)
    overall_pass = bool(leak_pass and in_task_pass and transfer_pass and topo_pass)

    report_lines.append("## Gate status")
    report_lines.append(f"- leak_gate_pass={leak_pass}")
    report_lines.append(f"- in_task_pass={in_task_pass}")
    report_lines.append(f"- transfer_pass={transfer_pass} (n_train_gate={n_train_gate})")
    report_lines.append(f"- topology_pass={topo_pass} (expect PR diff < 0 or mass diff > 0)")
    report_lines.append(f"- overall_pass={overall_pass}")
    report_lines.append("")

    report_lines.append("## Why v3 answers the NHSEA hypothesis better than v1/v2")
    report_lines.append(
        "- v1/v2 transfer sat at chance, so reciprocity tests were non-discriminative."
    )
    report_lines.append(
        "- v3 enforces an explicit OBC vs PBC topology contrast tied to boundary localization observables."
    )
    report_lines.append(
        "- Baseline transfer-learnability is gated before any mechanism compute is allowed."
    )

    report_path = Path(args.report)
    report_path.write_text("\n".join(report_lines) + "\n")

    report_json = {
        "leak_gate": {
            "forward": leak_fwd_data,
            "backward": leak_bwd_data,
        },
        "in_task": {
            "conclusion": conc_metrics,
            "topology": topo_metrics,
        },
        "zero_shot": {
            "conclusion_to_topology": zf_metrics,
            "topology_to_conclusion": zb_metrics,
        },
        "few_shot": fewshot_metrics,
        "topology": {
            "pr_mean": pr_mean,
            "pr_ci_low": pr_ci[0],
            "pr_ci_high": pr_ci[1],
            "mass_mean": mass_mean,
            "mass_ci_low": mass_ci[0],
            "mass_ci_high": mass_ci[1],
            "n_pairs": len(pr_diffs),
        },
        "gates": {
            "leak_gate_pass": leak_pass,
            "in_task_pass": in_task_pass,
            "transfer_pass": transfer_pass,
            "topology_pass": topo_pass,
            "overall_pass": overall_pass,
            "n_train_gate": n_train_gate,
        },
    }

    json_path = Path(args.json)
    json_path.write_text(json.dumps(report_json, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {out_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
