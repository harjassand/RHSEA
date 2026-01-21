"""Synthetic generators for NHSEA v2 invertible-mapping tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .seeding import instance_seed_u32


PAD_TOKEN = "PAD"
SEP_TOKEN = "SEP"
QRY_TOKEN = "QRY"
TASK_FWD_TOKEN = "TASK_FWD"
TASK_BWD_TOKEN = "TASK_BWD"


@dataclass(frozen=True)
class V2MappingConfig:
    n_symbols: int = 16
    n_facts: int = 8
    T: int = 64
    vocab_size: int = 200


@dataclass(frozen=True)
class V2MappingInstance:
    instance_id: str
    run_id: str
    seed: int
    task: str
    tokens: List[str]
    fact_pairs: List[Tuple[str, str]]
    fact_spans: List[Tuple[int, int]]
    query_token: str
    query_index: int
    mapping: List[int]
    candidates: Tuple[str, str]
    candidate_spans: Tuple[Tuple[int, int], Tuple[int, int]]
    true_index: int


def _symbol_token(prefix: str, idx: int) -> str:
    return f"{prefix}{idx:02d}"


def vocab_tokens_v2(n_symbols: int = 16, vocab_size: int = 200) -> List[str]:
    tokens = [PAD_TOKEN, SEP_TOKEN, QRY_TOKEN, TASK_FWD_TOKEN, TASK_BWD_TOKEN]
    tokens.extend([_symbol_token("A", i) for i in range(n_symbols)])
    tokens.extend([_symbol_token("B", i) for i in range(n_symbols)])
    tokens.extend([f"w{i}" for i in range(vocab_size)])
    return tokens


def build_vocab_v2(n_symbols: int = 16, vocab_size: int = 200) -> Dict[str, int]:
    tokens = vocab_tokens_v2(n_symbols=n_symbols, vocab_size=vocab_size)
    return {tok: i for i, tok in enumerate(tokens)}


def encode_tokens(tokens: Sequence[str], vocab: Dict[str, int]) -> List[int]:
    return [vocab[tok] for tok in tokens]


def _bijection(rng: np.random.Generator, n_symbols: int) -> List[int]:
    return list(rng.permutation(n_symbols))


def generate_v2_mapping(
    run_id: str,
    instance_id: str,
    cfg: V2MappingConfig,
    task: str,
) -> V2MappingInstance:
    if cfg.n_facts > cfg.n_symbols:
        raise ValueError("n_facts must be <= n_symbols")
    if task not in ("forward", "backward"):
        raise ValueError("task must be 'forward' or 'backward'")

    seed = instance_seed_u32(run_id, instance_id, salt="GEN_V2")
    rng = np.random.default_rng(seed)

    mapping = _bijection(rng, cfg.n_symbols)
    fact_ids = rng.choice(cfg.n_symbols, size=cfg.n_facts, replace=False)

    tokens: List[str] = [TASK_FWD_TOKEN if task == "forward" else TASK_BWD_TOKEN]
    fact_spans: List[Tuple[int, int]] = []
    fact_pairs: List[Tuple[str, str]] = []
    for idx in fact_ids:
        a_tok = _symbol_token("A", int(idx))
        b_tok = _symbol_token("B", int(mapping[int(idx)]))
        start = len(tokens)
        tokens.extend([a_tok, SEP_TOKEN, b_tok])
        end = len(tokens) - 1
        fact_spans.append((start, end))
        fact_pairs.append((a_tok, b_tok))

    query_index = int(rng.choice(fact_ids))
    if task == "forward":
        query_token = _symbol_token("A", query_index)
        true_token = _symbol_token("B", mapping[query_index])
        decoy_pool = [i for i in range(cfg.n_symbols) if i != mapping[query_index]]
        decoy_token = _symbol_token("B", int(rng.choice(decoy_pool)))
    else:
        query_token = _symbol_token("B", mapping[query_index])
        true_token = _symbol_token("A", query_index)
        decoy_pool = [i for i in range(cfg.n_symbols) if i != query_index]
        decoy_token = _symbol_token("A", int(rng.choice(decoy_pool)))

    tokens.extend([QRY_TOKEN, query_token])
    if len(tokens) > cfg.T - 2:
        raise ValueError("Sequence too long for fixed candidate slots")

    pad_needed = cfg.T - 2 - len(tokens)
    if pad_needed > 0:
        tokens.extend([PAD_TOKEN] * pad_needed)

    if rng.random() < 0.5:
        candidates = (true_token, decoy_token)
        true_index = 0
    else:
        candidates = (decoy_token, true_token)
        true_index = 1

    cand0_pos = cfg.T - 2
    cand1_pos = cfg.T - 1
    tokens.append(candidates[0])
    tokens.append(candidates[1])
    candidate_spans = ((cand0_pos, cand0_pos), (cand1_pos, cand1_pos))

    return V2MappingInstance(
        instance_id=instance_id,
        run_id=run_id,
        seed=seed,
        task=task,
        tokens=tokens,
        fact_pairs=fact_pairs,
        fact_spans=fact_spans,
        query_token=query_token,
        query_index=query_index,
        mapping=mapping,
        candidates=candidates,
        candidate_spans=candidate_spans,
        true_index=true_index,
    )
