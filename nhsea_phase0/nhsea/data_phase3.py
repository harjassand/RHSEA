"""Datasets for Phase 3 reciprocity tasks."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch.utils.data import Dataset

from .data import DatasetConfig
from .generators import build_vocab, encode_tokens
from .generators_phase3 import Phase3ChainConfig, generate_phase3_backward, generate_phase3_forward


class Phase3ForwardDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, gen_cfg: Phase3ChainConfig) -> None:
        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.vocab = build_vocab(max_props=cfg.max_props, vocab_size=cfg.vocab_size)
        self.run_id = f"phase3_forward_{cfg.split}_seed{cfg.seed}"

    def __len__(self) -> int:
        return self.cfg.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        instance_id = f"{self.cfg.split}_{idx}"
        inst = generate_phase3_forward(self.run_id, instance_id, self.gen_cfg)
        input_ids = encode_tokens(inst.tokens, self.vocab)
        attn_mask = [1 if tok != 0 else 0 for tok in input_ids]
        label = inst.true_index
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "meta": inst,
        }


class Phase3BackwardDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, gen_cfg: Phase3ChainConfig) -> None:
        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.vocab = build_vocab(max_props=cfg.max_props, vocab_size=cfg.vocab_size)
        self.run_id = f"phase3_backward_{cfg.split}_seed{cfg.seed}"

    def __len__(self) -> int:
        return self.cfg.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        instance_id = f"{self.cfg.split}_{idx}"
        inst = generate_phase3_backward(self.run_id, instance_id, self.gen_cfg)
        input_ids = encode_tokens(inst.tokens, self.vocab)
        attn_mask = [1 if tok != 0 else 0 for tok in input_ids]
        label = inst.true_index
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "meta": inst,
        }
