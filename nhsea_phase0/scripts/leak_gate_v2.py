#!/usr/bin/env python
"""NHSEA v2: generator-only leak gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from nhsea.generators_v2 import V2MappingConfig
from nhsea.leak_gate_v2 import run_leak_gate, write_leak_gate_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["forward", "backward"], required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_id", type=str, default="v2")
    ap.add_argument("--auroc_max", type=float, default=0.55)
    ap.add_argument("--report", type=str, default="runs/v2/leak_gate_v2_report.json")
    ap.add_argument("--features_out", type=str, default="runs/v2/leak_gate_v2_features.json")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--n_symbols", type=int, default=16)
    ap.add_argument("--n_facts", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--vocab_size", type=int, default=200)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = V2MappingConfig(
        n_symbols=args.n_symbols,
        n_facts=args.n_facts,
        T=args.T,
        vocab_size=args.vocab_size,
    )

    report, features = run_leak_gate(
        task=args.task,
        n=args.n,
        seed=args.seed,
        run_id=args.run_id,
        auroc_max=args.auroc_max,
        bootstrap=args.bootstrap,
        ci_alpha=args.ci_alpha,
        cfg=cfg,
    )
    report_path = Path(args.report)
    features_path = Path(args.features_out)
    write_leak_gate_report(report, report_path, features, features_path)

    if args.verbose:
        print("V2MappingConfig:", asdict(cfg))
        print("AUROC:", report["auroc"])

    if report["auroc"] > args.auroc_max:
        print(f"FAIL leak gate ({args.task}): AUROC={report['auroc']:.4f} > {args.auroc_max}")
        return 2
    print(f"PASS leak gate ({args.task}): AUROC={report['auroc']:.4f} <= {args.auroc_max}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
