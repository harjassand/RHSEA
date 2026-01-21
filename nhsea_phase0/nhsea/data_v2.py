"""Datasets and collators for NHSEA v2 tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .generators_v2 import V2MappingConfig, build_vocab_v2, encode_tokens, generate_v2_mapping


@dataclass(frozen=True)
class V2DatasetConfig:
    task: str
    split: str
    size: int
    seed: int
    T: int = 64
    n_symbols: int = 16
    n_facts: int = 8
    vocab_size: int = 200


class V2MappingDataset(Dataset):
    def __init__(self, cfg: V2DatasetConfig, gen_cfg: V2MappingConfig) -> None:
        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.vocab = build_vocab_v2(n_symbols=cfg.n_symbols, vocab_size=cfg.vocab_size)
        self.run_id = f"v2_{cfg.task}_{cfg.split}_seed{cfg.seed}"

    def __len__(self) -> int:
        return self.cfg.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        instance_id = f"{self.cfg.split}_{idx}"
        inst = generate_v2_mapping(self.run_id, instance_id, self.gen_cfg, self.cfg.task)
        input_ids = encode_tokens(inst.tokens, self.vocab)
        attn_mask = [1 if tok != 0 else 0 for tok in input_ids]
        label = inst.true_index
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
