#!/usr/bin/env python
"""Aggregate U-symmetry drift logs into a markdown report and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


def _parse_run_id(run_id: str) -> Dict[str, str] | None:
    match = re.match(
        r"^phase2_v2_forward_(?P<variant>mechanism|symmetric_control_v2_normmatched|no_injection|no_drift)_seed(?P<seed>\d+)$",
        run_id,
    )
    if not match:
        return None
    return match.groupdict()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs/phase2_claimA_v2")
    ap.add_argument("--out_md", type=str, default="")
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    root = Path(args.root)
    rows: List[Dict[str, str]] = []

    for drift_path in root.glob("**/u_symmetry_drift.jsonl"):
        run_dir = drift_path.parent
        run_id = run_dir.name
        meta = _parse_run_id(run_id)
        if meta is None:
            continue
        with drift_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rows.append(
                    {
                        "run_id": run_id,
                        "variant": meta["variant"],
                        "seed": meta["seed"],
                        "label": rec.get("label", ""),
                        "step": str(rec.get("step", "")),
                        "ratio_mean": str(rec.get("ratio_mean", "")),
                        "ratio_median": str(rec.get("ratio_median", "")),
                    }
                )

    out_csv = Path(args.out_csv) if args.out_csv else root / "u_symmetry_drift.csv"
    out_md = Path(args.out_md) if args.out_md else root / "u_symmetry_drift.md"

    if rows:
        keys = ["run_id", "variant", "seed", "label", "step", "ratio_mean", "ratio_median"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        lines = [
            "# U Symmetry Drift",
            "",
            f"root: {root}",
            "",
            "| run_id | variant | seed | label | step | ratio_mean | ratio_median |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            lines.append(
                f"| {row['run_id']} | {row['variant']} | {row['seed']} | {row['label']} | "
                f"{row['step']} | {row['ratio_mean']} | {row['ratio_median']} |"
            )
        out_md.write_text("\n".join(lines) + "\n")
    else:
        out_csv.write_text("")
        out_md.write_text("# U Symmetry Drift\n\n(no records found)\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
