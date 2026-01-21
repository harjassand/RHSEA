#!/usr/bin/env python
"""Aggregate NHSEA v2 baseline preflight runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--report", type=str, required=True)
    args = ap.parse_args()

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

    report_lines = ["# V2 Preflight Report", ""]

    leak_fwd = root / "leak_gate_v2_forward.json"
    leak_bwd = root / "leak_gate_v2_backward.json"
    leak_fwd_data = _load_json(leak_fwd) if leak_fwd.exists() else None
    leak_bwd_data = _load_json(leak_bwd) if leak_bwd.exists() else None

    report_lines.append("## Leak gate")
    if leak_fwd_data:
        report_lines.append(
            f"- forward: AUROC={leak_fwd_data['auroc']:.4f} pass={leak_fwd_data['passed']}"
        )
    if leak_bwd_data:
        report_lines.append(
            f"- backward: AUROC={leak_bwd_data['auroc']:.4f} pass={leak_bwd_data['passed']}"
        )
    report_lines.append("")

    def _acc_line(label: str, rows_subset: List[dict]) -> str:
        correct, total = _sum_counts(rows_subset)
        acc = correct / total if total else 0.0
        ci_low, ci_high = wilson_ci(correct, total)
        return f"- {label}: acc={acc:.4f} CI=[{ci_low:.4f},{ci_high:.4f}] n={total}"

    report_lines.append("## In-task accuracy")
    report_lines.append(_acc_line("forward", _filter("in_task", "forward", "forward")))
    report_lines.append(_acc_line("backward", _filter("in_task", "backward", "backward")))
    report_lines.append("")

    report_lines.append("## Zero-shot transfer")
    report_lines.append(_acc_line("forward→backward", _filter("zero_shot", "forward", "backward")))
    report_lines.append(_acc_line("backward→forward", _filter("zero_shot", "backward", "forward")))
    report_lines.append("")

    report_lines.append("## Few-shot transfer (head-only)")
    for n_train in (32, 128, 512):
        report_lines.append(_acc_line(f"forward→backward n={n_train}", _filter("few_shot", "forward", "backward", n_train)))
        report_lines.append(_acc_line(f"backward→forward n={n_train}", _filter("few_shot", "backward", "forward", n_train)))
    report_lines.append("")

    # Gate checks
    def _pass_in_task() -> bool:
        for task in ("forward", "backward"):
            rows_subset = _filter("in_task", task, task)
            correct, total = _sum_counts(rows_subset)
            acc = correct / total if total else 0.0
            if acc < 0.95:
                return False
        return True

    def _pass_zero_shot() -> bool:
        for train_task, eval_task in (("forward", "backward"), ("backward", "forward")):
            rows_subset = _filter("zero_shot", train_task, eval_task)
            correct, total = _sum_counts(rows_subset)
            acc = correct / total if total else 0.0
            if acc < 0.60:
                return False
        return True

    def _pass_few_shot_512() -> bool:
        for train_task, eval_task in (("forward", "backward"), ("backward", "forward")):
            rows_subset = _filter("few_shot", train_task, eval_task, 512)
            correct, total = _sum_counts(rows_subset)
            acc = correct / total if total else 0.0
            if acc < 0.70:
                return False
        return True

    leak_pass = bool(leak_fwd_data and leak_bwd_data and leak_fwd_data.get("passed") and leak_bwd_data.get("passed"))
    in_task_pass = _pass_in_task()
    transfer_pass = _pass_zero_shot() or _pass_few_shot_512()
    overall_pass = leak_pass and in_task_pass and transfer_pass

    report_lines.append("## Gate status")
    report_lines.append(f"- leak_gate_pass={leak_pass}")
    report_lines.append(f"- in_task_pass={in_task_pass}")
    report_lines.append(f"- transfer_pass={transfer_pass}")
    report_lines.append(f"- overall_pass={overall_pass}")

    report_path = Path(args.report)
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {out_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
