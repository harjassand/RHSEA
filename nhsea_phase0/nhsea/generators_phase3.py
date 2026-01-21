"""Phase 3 generators for forward/abductive reciprocity tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .generators import (
    ForwardChainInstance,
    BackwardChainInstance,
    _append_edges,
    _place_props_sequential,
)
from .seeding import instance_seed_u32


@dataclass(frozen=True)
class Phase3ChainConfig:
    T: int = 64
    n_mid: int = 2
    prop_len_min: int = 3
    prop_len_max: int = 6
    vocab_size: int = 200


def _base_run_id(run_id: str) -> str:
    out = run_id.replace("forward", "paired")
    out = out.replace("backward", "paired")
    return out


def _phase3_base(
    run_id: str,
    instance_id: str,
    cfg: Phase3ChainConfig,
) -> Tuple[List[str], List[Tuple[int, int]], Tuple[int, int], Tuple[int, int], int, int, int]:
    if cfg.n_mid < 2:
        raise ValueError("Phase3ChainConfig.n_mid must be >= 2")
    base_id = _base_run_id(run_id)
    seed = instance_seed_u32(base_id, instance_id, salt="GEN_P3_BASE")
    rng = np.random.default_rng(seed)

    # IDs: antecedents (2), mid nodes (n_mid), conclusions (2), filler premise (1)
    a1, a2 = 0, 1
    mid_start = 2
    mid_ids = list(range(mid_start, mid_start + cfg.n_mid))
    c1 = mid_start + cfg.n_mid
    c2 = c1 + 1
    filler = c2 + 1

    prop_ids = list(range(filler + 1))
    lengths = [int(rng.integers(cfg.prop_len_min, cfg.prop_len_max + 1)) for _ in prop_ids]

    cand_len = int(rng.integers(cfg.prop_len_min, cfg.prop_len_max + 1))
    lengths[a1] = cand_len
    lengths[a2] = cand_len
    concl_len = int(rng.integers(cfg.prop_len_min, cfg.prop_len_max + 1))
    lengths[c1] = concl_len
    lengths[c2] = concl_len

    tokens, prop_spans = _place_props_sequential(rng, prop_ids, lengths, cfg.vocab_size)

    true_ante_idx = int(rng.integers(0, 2))
    true_concl_idx = int(rng.integers(0, 2))
    antecedents = (a1, a2)
    conclusions = (c1, c2)
    a_true = antecedents[true_ante_idx]
    a_false = antecedents[1 - true_ante_idx]
    c_true = conclusions[true_concl_idx]
    c_false = conclusions[1 - true_concl_idx]

    n_true = max(1, cfg.n_mid // 2)
    n_decoy = max(1, cfg.n_mid - n_true)
    if n_true + n_decoy > cfg.n_mid:
        n_true = cfg.n_mid - n_decoy
    mid_true = mid_ids[:n_true]
    mid_decoy = mid_ids[n_true : n_true + n_decoy]

    edges: List[Tuple[int, int]] = []
    for mid in mid_true:
        edges.append((a_true, mid))
        edges.append((mid, c_true))
    for mid in mid_decoy:
        edges.append((a_false, mid))
        edges.append((mid, c_false))

    tokens = _append_edges(tokens, edges, cfg.T)
    return tokens, prop_spans, antecedents, conclusions, filler, true_ante_idx, true_concl_idx


def generate_phase3_forward(run_id: str, instance_id: str, cfg: Phase3ChainConfig) -> ForwardChainInstance:
    tokens, prop_spans, antecedents, conclusions, filler, true_ante_idx, true_concl_idx = _phase3_base(
        run_id, instance_id, cfg
    )
    premises = [antecedents[true_ante_idx], filler]
    return ForwardChainInstance(
        instance_id=instance_id,
        tokens=tokens,
        prop_spans=prop_spans,
        premises=premises,
        candidates=conclusions,
        true_index=true_concl_idx,
    )


def generate_phase3_backward(run_id: str, instance_id: str, cfg: Phase3ChainConfig) -> BackwardChainInstance:
    tokens, prop_spans, antecedents, _conclusions, _filler, true_ante_idx, _true_concl_idx = _phase3_base(
        run_id, instance_id, cfg
    )
    return BackwardChainInstance(
        instance_id=instance_id,
        tokens=tokens,
        prop_spans=prop_spans,
        candidates=antecedents,
        true_index=true_ante_idx,
    )
