"""Synthetic generators (Phase 0 skeleton).

This is a minimal, testable generator implementation to support:
- Phase 0.1 generator-only leak checks for the forward-chain decoy task.
- Phase 0.2/0.3 unit tests and CLI wiring.

The design goal is determinism and symmetry between the two candidate conclusions.
It is not intended to be the final research generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .seeding import instance_seed_u32


@dataclass(frozen=True)
class ForwardChainInstance:
    instance_id: str
    tokens: List[str]
    # proposition spans: list of (start, end) inclusive indices in tokens
    prop_spans: List[Tuple[int, int]]
    # premise proposition IDs
    premises: List[int]
    # two candidate proposition IDs
    candidates: Tuple[int, int]
    # index in candidates tuple which is true (0 or 1)
    true_index: int


@dataclass(frozen=True)
class ForwardChainConfig:
    n_prem: int = 3
    cand_len: int = 3
    vocab_size: int = 200


def _rand_word(rng: np.random.Generator, vocab_size: int) -> str:
    return f"w{int(rng.integers(0, vocab_size))}"


def generate_forward_chain(run_id: str, instance_id: str, cfg: ForwardChainConfig) -> ForwardChainInstance:
    """Generate a forward-chain instance with two symmetric candidates.

    Construction:
    - Build n_prem premise propositions, each a fixed-length phrase.
    - Build two candidate propositions with identical length distribution.
    - Randomly choose which candidate is true with p=0.5 (deterministic given seed).

    Proposition IDs are 0..M-1 in the order spans are appended.
    """
    seed = instance_seed_u32(run_id, instance_id, salt="GEN_V1")
    rng = np.random.default_rng(seed)

    tokens: List[str] = []
    prop_spans: List[Tuple[int, int]] = []

    def add_prop(prefix: str, length: int) -> int:
        start = len(tokens)
        tokens.append(prefix)
        for _ in range(length - 1):
            tokens.append(_rand_word(rng, cfg.vocab_size))
        end = len(tokens) - 1
        prop_spans.append((start, end))
        return len(prop_spans) - 1

    # Premises
    premises = [add_prop("P:", cfg.cand_len) for _ in range(cfg.n_prem)]

    # Candidates: same template, same length
    cand_ids = (add_prop("C1:", cfg.cand_len), add_prop("C2:", cfg.cand_len))

    true_index = int(rng.integers(0, 2))

    return ForwardChainInstance(
        instance_id=instance_id,
        tokens=tokens,
        prop_spans=prop_spans,
        premises=premises,
        candidates=cand_ids,
        true_index=true_index,
    )


def candidate_token_spans(inst: ForwardChainInstance) -> Tuple[Sequence[int], Sequence[int]]:
    """Return token indices belonging to the two candidate spans."""
    c1, c2 = inst.candidates
    s1, e1 = inst.prop_spans[c1]
    s2, e2 = inst.prop_spans[c2]
    return list(range(s1, e1 + 1)), list(range(s2, e2 + 1))


def premise_token_set(inst: ForwardChainInstance) -> Sequence[int]:
    """Return token indices belonging to all premise spans."""
    idxs: List[int] = []
    for pid in inst.premises:
        s, e = inst.prop_spans[pid]
        idxs.extend(range(s, e + 1))
    return idxs
