#!/usr/bin/env python
"""Accuracy-only evaluation with optional task override."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nhsea.data import BackwardChainDataset, CycleRegimeDataset, DatasetConfig, ForwardChainDataset, collate_batch
from nhsea.data_phase3 import Phase3BackwardDataset, Phase3ForwardDataset
from nhsea.generators import BackwardChainConfig, CycleRegimeConfig, ForwardChainConfig
from nhsea.generators_phase3 import Phase3ChainConfig
from nhsea.model import ModelConfig, TinyTransformer


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--task", choices=["forward", "cycle", "backward"], required=True)
    ap.add_argument("--eval_size", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--phase3", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    if args.task == "cycle":
        model.set_num_classes(4)
    else:
        model.set_num_classes(2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    phase3 = bool(ckpt.get("phase3", False) or args.phase3)
    data_cfg = DatasetConfig(task=args.task, split="eval", size=args.eval_size, seed=int(ckpt["seed"]))
    if args.task == "forward":
        if phase3:
            gen_cfg = Phase3ChainConfig()
            dataset = Phase3ForwardDataset(data_cfg, gen_cfg)
        else:
            gen_cfg = ForwardChainConfig()
            dataset = ForwardChainDataset(data_cfg, gen_cfg)
    elif args.task == "cycle":
        gen_cfg = CycleRegimeConfig()
        dataset = CycleRegimeDataset(data_cfg, gen_cfg)
    else:
        if phase3:
            gen_cfg = Phase3ChainConfig()
            dataset = Phase3BackwardDataset(data_cfg, gen_cfg)
        else:
            gen_cfg = BackwardChainConfig()
            dataset = BackwardChainDataset(data_cfg, gen_cfg)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    all_preds = []
    all_labels = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inst_path = out_dir / "eval_instances.jsonl.gz"
    inst_f = gzip.open(inst_path, "wt", encoding="utf-8")

    with torch.no_grad():
        for input_ids, attn_mask, labels, metas in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            logits, _, _ = model(input_ids, attn_mask=attn_mask, variant=ckpt["variant"], alpha=ckpt["alpha"], beta=ckpt["beta"])
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())
            for i, meta in enumerate(metas):
                record = {
                    "instance_id": meta.instance_id,
                    "label": int(labels[i].item()),
                    "pred": int(preds[i]),
                }
                inst_f.write(json.dumps(record) + "\n")

    inst_f.close()
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    summary = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "accuracy": accuracy,
        "phase3": phase3,
        "git_commit": _git_commit(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "summary.csv").write_text("summary\n" + json.dumps(summary))
    (out_dir / "eval_config.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "task": args.task,
                "eval_size": args.eval_size,
                "batch_size": args.batch_size,
                "phase3": phase3,
                "git_commit": _git_commit(),
                "gen_cfg": gen_cfg.__dict__,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
