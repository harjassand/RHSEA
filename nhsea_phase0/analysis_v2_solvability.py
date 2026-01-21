#!/usr/bin/env python
"""Audit logical solvability of NHSEA v2 generator (no model)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from nhsea.generators_v2 import V2MappingConfig, generate_v2_mapping


def _parse_seeds(raw: str) -> List[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def _load_gen_cfg_from_checkpoint(path: Path) -> V2MappingConfig:
    ckpt = torch.load(path, map_location="cpu")
    return V2MappingConfig(**ckpt["gen_cfg"])


def _resolve_gen_cfg(checkpoint: str, run_root: str) -> Tuple[V2MappingConfig, str]:
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return _load_gen_cfg_from_checkpoint(ckpt_path), str(ckpt_path)

    root = Path(run_root)
    fallback = root / "forward_train" / "no_injection" / "seed_0" / "checkpoint_final.pt"
    if fallback.exists():
        return _load_gen_cfg_from_checkpoint(fallback), str(fallback)

    return V2MappingConfig(), "default_config"


def _is_ambiguous(inst) -> Tuple[bool, str]:
    if inst.task == "forward":
        fact_map = {a: b for a, b in inst.fact_pairs}
        expected = fact_map.get(inst.query_token)
    else:
        fact_map = {b: a for a, b in inst.fact_pairs}
        expected = fact_map.get(inst.query_token)

    if expected is None:
        return True, "missing_query_in_facts"

    matches = [i for i, cand in enumerate(inst.candidates) if cand == expected]
    if len(matches) != 1:
        return True, "candidate_mismatch"
    return False, ""


def audit_task(task: str, cfg: V2MappingConfig, seeds: List[int], eval_size: int) -> Dict[str, object]:
    total = 0
    ambiguous = 0
    reasons: Dict[str, int] = {}
    by_seed = []
    for seed in seeds:
        run_id = f"v2_{task}_eval_seed{seed}"
        seed_total = 0
        seed_ambiguous = 0
        for i in range(eval_size):
            instance_id = f"eval_{i}"
            inst = generate_v2_mapping(run_id, instance_id, cfg, task)
            is_ambig, reason = _is_ambiguous(inst)
            total += 1
            seed_total += 1
            if is_ambig:
                ambiguous += 1
                seed_ambiguous += 1
                reasons[reason] = reasons.get(reason, 0) + 1
        by_seed.append(
            {
                "seed": seed,
                "total": seed_total,
                "ambiguous": seed_ambiguous,
                "ambiguous_rate": seed_ambiguous / seed_total if seed_total else 0.0,
            }
        )

    ambiguous_rate = ambiguous / total if total else 0.0
    ceiling_acc = 1.0 - 0.5 * ambiguous_rate
    return {
        "task": task,
        "total": total,
        "ambiguous": ambiguous,
        "ambiguous_rate": ambiguous_rate,
        "ceiling_acc": ceiling_acc,
        "reasons": reasons,
        "by_seed": by_seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=str, default="runs/v2_preflight")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--eval_size", type=int, default=10000)
    ap.add_argument("--out_json", type=str, default="runs/v2_preflight/solvability_audit.json")
    ap.add_argument("--out_md", type=str, default="runs/v2_preflight/solvability_audit.md")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    cfg, cfg_source = _resolve_gen_cfg(args.checkpoint, args.run_root)

    forward_stats = audit_task("forward", cfg, seeds, args.eval_size)
    backward_stats = audit_task("backward", cfg, seeds, args.eval_size)

    total = forward_stats["total"] + backward_stats["total"]
    ambiguous = forward_stats["ambiguous"] + backward_stats["ambiguous"]
    ambiguous_rate = ambiguous / total if total else 0.0
    ceiling_acc = 1.0 - 0.5 * ambiguous_rate

    report = {
        "config_source": cfg_source,
        "config": asdict(cfg),
        "seeds": seeds,
        "eval_size": args.eval_size,
        "forward": forward_stats,
        "backward": backward_stats,
        "overall": {
            "total": total,
            "ambiguous": ambiguous,
            "ambiguous_rate": ambiguous_rate,
            "ceiling_acc": ceiling_acc,
        },
        "by_n_facts": {
            str(cfg.n_facts): {
                "ambiguous_rate": ambiguous_rate,
                "ceiling_acc": ceiling_acc,
                "total": total,
            }
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# V2 Solvability Audit",
        "",
        "This audit uses only generator logic; no model is involved.",
        "",
        f"Config source: {cfg_source}",
        f"Config: n_symbols={cfg.n_symbols}, n_facts={cfg.n_facts}, T={cfg.T}, vocab_size={cfg.vocab_size}",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        f"Eval size per seed/task: {args.eval_size}",
        "",
        "## Summary",
        f"- forward ambiguous_rate={forward_stats['ambiguous_rate']:.6f}, ceiling_acc={forward_stats['ceiling_acc']:.6f}",
        f"- backward ambiguous_rate={backward_stats['ambiguous_rate']:.6f}, ceiling_acc={backward_stats['ceiling_acc']:.6f}",
        f"- overall ambiguous_rate={ambiguous_rate:.6f}, ceiling_acc={ceiling_acc:.6f}",
        "",
        "## By n_facts",
        f"- n_facts={cfg.n_facts}: ambiguous_rate={ambiguous_rate:.6f}, ceiling_acc={ceiling_acc:.6f}",
    ]

    out_md = Path(args.out_md)
    out_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
