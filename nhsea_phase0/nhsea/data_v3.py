"""Datasets for NHSEA v3 baseline-only tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .generators_v3 import V3Config, generate_v3_instance


@dataclass(frozen=True)
class V3DatasetConfig:
    task: str
    split: str
    size: int
    seed: int
    T: int = 64
    M: int = 8
    K: int = 4
    L_min: int = 3
    L_max: int = 6
    vocab_size: int = 200


def vocab_tokens_v3(M: int = 8, vocab_size: int = 200) -> List[str]:
    tokens = [
        "PAD",
        "EDGE",
        "QRY",
        "TASK_CONC",
        "TASK_TOPO",
    ]
    tokens.extend([f"P{idx:02d}" for idx in range(M)])
    tokens.extend([f"w{idx}" for idx in range(vocab_size)])
    return tokens


def build_vocab_v3(M: int = 8, vocab_size: int = 200) -> Dict[str, int]:
    toks = vocab_tokens_v3(M=M, vocab_size=vocab_size)
    return {tok: i for i, tok in enumerate(toks)}


def encode_tokens(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    return [vocab[tok] for tok in tokens]


class V3Dataset(Dataset):
    def __init__(self, cfg: V3DatasetConfig, gen_cfg: V3Config) -> None:
        if cfg.size % 2 != 0:
            raise ValueError("size must be even to preserve OBC/PBC pairing")
        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.vocab = build_vocab_v3(M=cfg.M, vocab_size=cfg.vocab_size)
        self.run_id = f"v3_{cfg.task}_{cfg.split}_seed{cfg.seed}"

    def __len__(self) -> int:
        return self.cfg.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair_id = f"{self.cfg.split}_pair_{idx // 2}"
        topology = "OBC" if idx % 2 == 0 else "PBC"
        instance_id = pair_id
        inst = generate_v3_instance(
            run_id=self.run_id,
            instance_id=instance_id,
            pair_id=pair_id,
            topology=topology,
            task=self.cfg.task,
            cfg=self.gen_cfg,
        )
        input_ids = encode_tokens(inst.tokens, self.vocab)
        attn_mask = [1 if tok != 0 else 0 for tok in input_ids]

        if self.cfg.task == "conclusion":
            label = inst.true_index
        elif self.cfg.task == "topology":
            label = 0 if inst.topology == "OBC" else 1
        else:
            raise ValueError(f"Unknown task: {self.cfg.task}")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "meta": inst,
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Any]]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attn_mask = torch.stack([b["attn_mask"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    metas = [b["meta"] for b in batch]
    return input_ids, attn_mask, labels, metas
