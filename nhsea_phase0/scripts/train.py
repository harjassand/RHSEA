#!/usr/bin/env python
"""Phase 1 training entrypoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from nhsea.data import CycleRegimeDataset, DatasetConfig, ForwardChainDataset, collate_batch
from nhsea.generators import CycleRegimeConfig, ForwardChainConfig
from nhsea.model import ModelConfig, TinyTransformer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["forward", "cycle"], required=True)
    ap.add_argument("--variant", choices=["mechanism", "symmetric_control", "no_injection", "no_drift"], required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_size", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--out_dir", type=str, default="runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DatasetConfig(task=args.task, split="train", size=args.train_size, seed=args.seed)
    if args.task == "forward":
        gen_cfg = ForwardChainConfig()
        dataset = ForwardChainDataset(data_cfg, gen_cfg)
        num_classes = 2
    else:
        gen_cfg = CycleRegimeConfig()
        dataset = CycleRegimeDataset(data_cfg, gen_cfg)
        num_classes = 4

    model_cfg = ModelConfig(vocab_size=len(dataset.vocab))
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(num_classes)
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(args.seed),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    step = 0
    while step < args.steps:
        for input_ids, attn_mask, labels, _ in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            logits, _, _ = model(
                input_ids,
                attn_mask=attn_mask,
                variant=args.variant,
                alpha=args.alpha,
                beta=args.beta,
            )
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if step % 500 == 0:
                print(f"step {step} loss {loss.item():.4f}")
            if step >= args.steps:
                break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.task}_{args.variant}_seed{args.seed}"
    ckpt_path = out_dir / f"{run_name}.pt"
    config_path = out_dir / f"{run_name}.json"

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_cfg": asdict(model_cfg),
            "variant": args.variant,
            "task": args.task,
            "alpha": args.alpha,
            "beta": args.beta,
            "seed": args.seed,
            "vocab": dataset.vocab,
        },
        ckpt_path,
    )
    config_path.write_text(
        json.dumps(
            {
                "task": args.task,
                "variant": args.variant,
                "alpha": args.alpha,
                "beta": args.beta,
                "seed": args.seed,
                "train_size": args.train_size,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(f"Saved checkpoint: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
