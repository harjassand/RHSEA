#!/usr/bin/env python
"""Phase 3 multitask training: forward + backward with task-id token."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from nhsea.data import DatasetConfig, collate_batch
from nhsea.generators import build_vocab, encode_tokens
from nhsea.generators_phase3 import Phase3ChainConfig, generate_phase3_backward, generate_phase3_forward
from nhsea.model import ModelConfig, TinyTransformer


TASK_FWD = "TASK_FWD"
TASK_BWD = "TASK_BWD"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _build_task_vocab(max_props: int, vocab_size: int) -> Dict[str, int]:
    vocab = build_vocab(max_props=max_props, vocab_size=vocab_size)
    if TASK_FWD in vocab or TASK_BWD in vocab:
        raise ValueError("Task tokens already exist in vocab")
    vocab[TASK_FWD] = len(vocab)
    vocab[TASK_BWD] = len(vocab)
    return vocab


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps":
        return torch.device("mps")
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {name}")


class Phase3TaskDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, gen_cfg: Phase3ChainConfig, task: str, vocab: Dict[str, int]) -> None:
        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.task = task
        self.vocab = vocab
        self.run_id = f"phase3_{task}_{cfg.split}_seed{cfg.seed}"
        self.task_token = TASK_FWD if task == "forward" else TASK_BWD

    def __len__(self) -> int:
        return self.cfg.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        instance_id = f"{self.cfg.split}_{idx}"
        if self.task == "forward":
            inst = generate_phase3_forward(self.run_id, instance_id, self.gen_cfg)
        else:
            inst = generate_phase3_backward(self.run_id, instance_id, self.gen_cfg)
        input_ids = encode_tokens(inst.tokens, self.vocab)
        input_ids[0] = self.vocab[self.task_token]
        attn_mask = [1 if tok != 0 else 0 for tok in input_ids]
        label = inst.true_index
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "meta": inst,
        }


class Phase3MixedDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, gen_cfg: Phase3ChainConfig, vocab: Dict[str, int]) -> None:
        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.vocab = vocab
        self.run_id_fwd = f"phase3_forward_{cfg.split}_seed{cfg.seed}"
        self.run_id_bwd = f"phase3_backward_{cfg.split}_seed{cfg.seed}"

    def __len__(self) -> int:
        return self.cfg.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair_idx = idx // 2
        if idx % 2 == 0:
            inst = generate_phase3_forward(self.run_id_fwd, f"{self.cfg.split}_{pair_idx}", self.gen_cfg)
            task_token = TASK_FWD
        else:
            inst = generate_phase3_backward(self.run_id_bwd, f"{self.cfg.split}_{pair_idx}", self.gen_cfg)
            task_token = TASK_BWD
        input_ids = encode_tokens(inst.tokens, self.vocab)
        input_ids[0] = self.vocab[task_token]
        attn_mask = [1 if tok != 0 else 0 for tok in input_ids]
        label = inst.true_index
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "meta": inst,
        }


def _epoch_permutation(size: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(seed + epoch)
    return rng.permutation(size)


def _train(
    model: TinyTransformer,
    dataset: Dataset,
    cfg: argparse.Namespace,
    out_dir: Path,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    device = _resolve_device(cfg.device)
    model.to(device)
    model.train()

    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    dataset_size = len(dataset)
    batches_per_epoch = (dataset_size + cfg.batch_size - 1) // cfg.batch_size
    perm = _epoch_permutation(dataset_size, cfg.seed, 0)
    epoch = 0
    batch_idx = 0
    step = 0

    def _next_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal epoch, batch_idx, perm
        if batch_idx >= batches_per_epoch:
            epoch += 1
            batch_idx = 0
            perm = _epoch_permutation(dataset_size, cfg.seed, epoch)
        start = batch_idx * cfg.batch_size
        end = min(start + cfg.batch_size, dataset_size)
        idxs = perm[start:end]
        batch = [dataset[int(i)] for i in idxs]
        batch_idx += 1
        input_ids, attn_mask, labels, _ = collate_batch(batch)
        return input_ids, attn_mask, labels

    while step < cfg.steps:
        input_ids, attn_mask, labels = _next_batch()
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        logits, _, _ = model(
            input_ids,
            attn_mask=attn_mask,
            variant=cfg.variant,
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=1.0,
        )
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss encountered")
        opt.zero_grad()
        loss.backward()
        opt.step()

        step += 1
        if step % cfg.log_every == 0:
            preds = torch.argmax(logits, dim=-1)
            acc = float((preds == labels).float().mean().item())
            record = {"step": step, "loss": float(loss.item()), "acc": acc}
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

    ckpt = {
        "model_state": model.state_dict(),
        "model_cfg": asdict(ModelConfig(vocab_size=cfg.vocab_size)),
        "variant": cfg.variant,
        "task": "multitask",
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "seed": cfg.seed,
        "vocab": cfg.vocab,
        "git_commit": _git_commit(),
    }
    torch.save(ckpt, out_dir / "checkpoint_final.pt")


def _eval(
    model: TinyTransformer,
    dataset: Dataset,
    cfg: argparse.Namespace,
) -> float:
    device = _resolve_device(cfg.device)
    model.to(device)
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for i in range(0, len(dataset), cfg.batch_size):
            batch = [dataset[j] for j in range(i, min(i + cfg.batch_size, len(dataset)))]
            input_ids, attn_mask, labels, _ = collate_batch(batch)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            logits, _, _ = model(
                input_ids,
                attn_mask=attn_mask,
                variant=cfg.variant,
                alpha=cfg.alpha,
                beta=cfg.beta,
                gamma=1.0,
            )
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())
    return float(np.mean(np.array(all_preds) == np.array(all_labels)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["mechanism", "no_injection"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train_size", type=int, default=50000)
    ap.add_argument("--eval_size", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gen_cfg = Phase3ChainConfig()
    data_cfg = DatasetConfig(task="multitask", split="train", size=args.train_size, seed=args.seed)
    vocab = _build_task_vocab(max_props=data_cfg.max_props, vocab_size=data_cfg.vocab_size)

    mixed = Phase3MixedDataset(data_cfg, gen_cfg, vocab)
    model_cfg = ModelConfig(vocab_size=len(vocab))
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "seed": args.seed,
                "train_size": args.train_size,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "alpha": args.alpha,
                "beta": args.beta,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "model_cfg": asdict(model_cfg),
                "gen_cfg": asdict(gen_cfg),
                "git_commit": _git_commit(),
            },
            indent=2,
            sort_keys=True,
        )
    )

    cfg = argparse.Namespace(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        steps=args.steps,
        log_every=args.log_every,
        alpha=args.alpha,
        beta=args.beta,
        variant=args.variant,
        seed=args.seed,
        device=args.device,
        vocab=vocab,
        vocab_size=len(vocab),
    )

    ckpt = out_dir / "checkpoint_final.pt"
    if not ckpt.exists():
        _train(model, mixed, cfg, out_dir)
    else:
        ckpt_payload = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(ckpt_payload["model_state"])

    eval_cfg = DatasetConfig(task="multitask", split="eval", size=args.eval_size, seed=args.seed)
    eval_forward = Phase3TaskDataset(eval_cfg, gen_cfg, "forward", vocab)
    eval_backward = Phase3TaskDataset(eval_cfg, gen_cfg, "backward", vocab)
    acc_forward = _eval(model, eval_forward, cfg)
    acc_backward = _eval(model, eval_backward, cfg)

    (out_dir / "eval_forward.json").write_text(json.dumps({"accuracy": acc_forward}, indent=2))
    (out_dir / "eval_backward.json").write_text(json.dumps({"accuracy": acc_backward}, indent=2))
    print(f"Wrote {out_dir / 'eval_forward.json'}")
    print(f"Wrote {out_dir / 'eval_backward.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
