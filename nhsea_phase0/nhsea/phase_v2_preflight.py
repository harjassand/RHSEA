"""NHSEA v2 preflight smoke summary CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .generators_v2 import V2MappingConfig, generate_v2_mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["forward", "backward"], default="forward")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_id", type=str, default="v2_preflight")
    ap.add_argument("--out", type=str, default="runs/v2_preflight/smoke_summary.json")
    ap.add_argument("--n_symbols", type=int, default=16)
    ap.add_argument("--n_facts", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--vocab_size", type=int, default=200)
    args = ap.parse_args()

    cfg = V2MappingConfig(
        n_symbols=args.n_symbols,
        n_facts=args.n_facts,
        T=args.T,
        vocab_size=args.vocab_size,
    )

    labels = {"0": 0, "1": 0}
    candidate_positions = set()
    seeds = []
    run_key = f"{args.run_id}_{args.task}_seed{args.seed}"
    for i in range(args.n):
        inst = generate_v2_mapping(run_key, f"i{i}", cfg, args.task)
        labels[str(inst.true_index)] += 1
        candidate_positions.add(inst.candidate_spans[0][0])
        candidate_positions.add(inst.candidate_spans[1][0])
        seeds.append(inst.seed)

    summary = {
        "task": args.task,
        "run_id": run_key,
        "seed": args.seed,
        "n": args.n,
        "label_counts": labels,
        "candidate_positions": sorted(candidate_positions),
        "first_instance_seed": seeds[0] if seeds else None,
        "v2_gen": asdict(cfg),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
