"""Synthetic generators for Phase 1 tasks (forward-chain and cycle regimes)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .seeding import instance_seed_u32


PAD_TOKEN = "PAD"
EDGE_TOKEN = "E"


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
class CycleRegimeInstance:
    instance_id: str
    tokens: List[str]
    # proposition spans: list of (start, end) inclusive indices in tokens
    prop_spans: List[Tuple[int, int]]
    # premise proposition IDs
    premises: List[int]
    # regime label: 0 DAG, 1 C2, 2 C3, 3 C4
    regime: int


@dataclass(frozen=True)
class BackwardChainInstance:
    instance_id: str
    tokens: List[str]
    # proposition spans: list of (start, end) inclusive indices in tokens
    prop_spans: List[Tuple[int, int]]
    # candidate premise proposition IDs
    candidates: Tuple[int, int]
    # index in candidates tuple which is true (0 or 1)
    true_index: int

@dataclass(frozen=True)
class ForwardChainConfig:
    T: int = 64
    n_prem: int = 2
    n_mid: int = 2
    prop_len_min: int = 3
    prop_len_max: int = 6
    vocab_size: int = 200


@dataclass(frozen=True)
class BackwardChainConfig:
    T: int = 64
    n_cand_prem: int = 2
    n_mid: int = 2
    prop_len_min: int = 3
    prop_len_max: int = 6
    vocab_size: int = 200


@dataclass(frozen=True)
class CycleRegimeConfig:
    T: int = 64
    M: int = 8
    n_prem: int = 2
    prop_len_min: int = 3
    prop_len_max: int = 6
    vocab_size: int = 200
    edge_prob: float = 0.25


def vocab_tokens(max_props: int = 8, vocab_size: int = 200) -> List[str]:
    tokens = [PAD_TOKEN, EDGE_TOKEN]
    tokens.extend([f"P{idx:02d}" for idx in range(max_props)])
    tokens.extend([f"w{idx}" for idx in range(vocab_size)])
    return tokens


def build_vocab(max_props: int = 8, vocab_size: int = 200) -> Dict[str, int]:
    tokens = vocab_tokens(max_props=max_props, vocab_size=vocab_size)
    return {tok: i for i, tok in enumerate(tokens)}


def encode_tokens(tokens: Sequence[str], vocab: Dict[str, int]) -> List[int]:
    return [vocab[tok] for tok in tokens]


def _rand_word(rng: np.random.Generator, vocab_size: int) -> str:
    return f"w{int(rng.integers(0, vocab_size))}"


def _make_prop_tokens(rng: np.random.Generator, prop_id: int, length: int, vocab_size: int) -> List[str]:
    if length < 1:
        raise ValueError("length must be >= 1")
    toks = [f"P{prop_id:02d}"]
    for _ in range(length - 1):
        toks.append(_rand_word(rng, vocab_size))
    return toks


def _place_props_sequential(
    rng: np.random.Generator,
    prop_ids: Sequence[int],
    lengths: Sequence[int],
    vocab_size: int,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for pid, length in zip(prop_ids, lengths):
        start = len(tokens)
        tokens.extend(_make_prop_tokens(rng, pid, length, vocab_size))
        end = len(tokens) - 1
        spans.append((start, end))
    return tokens, spans


def _append_edges(tokens: List[str], edges: Sequence[Tuple[int, int]], T: int) -> List[str]:
    out = list(tokens)
    for a, b in edges:
        if len(out) + 3 > T:
            break
        out.extend([EDGE_TOKEN, f"P{a:02d}", f"P{b:02d}"])
    if len(out) > T:
        out = out[:T]
    if len(out) < T:
        out.extend([PAD_TOKEN] * (T - len(out)))
    return out


def _premise_token_ids(premises: Sequence[int]) -> List[int]:
    return list(premises)


def generate_forward_chain(run_id: str, instance_id: str, cfg: ForwardChainConfig) -> ForwardChainInstance:
    """Generate a forward-chain instance with anti-leak symmetry for candidates."""
    seed = instance_seed_u32(run_id, instance_id, salt="GEN_FWD_V1")
    rng = np.random.default_rng(seed)

    n_props = cfg.n_prem + cfg.n_mid + 2
    prop_ids = list(range(n_props))
    lengths = [int(rng.integers(cfg.prop_len_min, cfg.prop_len_max + 1)) for _ in prop_ids]
    cand_len = int(rng.integers(cfg.prop_len_min, cfg.prop_len_max + 1))
    lengths[-1] = cand_len
    lengths[-2] = cand_len

    tokens, prop_spans = _place_props_sequential(rng, prop_ids, lengths, cfg.vocab_size)

    # Premises are the first n_prem props; candidates are the last two props.
    premises = list(range(cfg.n_prem))
    candidates = (n_props - 2, n_props - 1)
    true_index = int(rng.integers(0, 2))
    true_cand = candidates[true_index]

    # Build a simple forward chain to the true candidate.
    edges: List[Tuple[int, int]] = []
    for i, prem in enumerate(premises):
        mid = cfg.n_prem + (i % cfg.n_mid)
        edges.append((prem, mid))
    for i in range(cfg.n_mid):
        edges.append((cfg.n_prem + i, true_cand))

    tokens = _append_edges(tokens, edges, cfg.T)

    return ForwardChainInstance(
        instance_id=instance_id,
        tokens=tokens,
        prop_spans=prop_spans,
        premises=premises,
        candidates=candidates,
        true_index=true_index,
    )


def generate_backward_chain(run_id: str, instance_id: str, cfg: BackwardChainConfig) -> BackwardChainInstance:
    """Generate a backward (abductive) chain instance with two candidate premises."""
    seed = instance_seed_u32(run_id, instance_id, salt="GEN_BWD_V1")
    rng = np.random.default_rng(seed)

    n_props = cfg.n_cand_prem + cfg.n_mid + 1
    prop_ids = list(range(n_props))
    lengths = [int(rng.integers(cfg.prop_len_min, cfg.prop_len_max + 1)) for _ in prop_ids]
    cand_len = int(rng.integers(cfg.prop_len_min, cfg.prop_len_max + 1))
    lengths[0] = cand_len
    lengths[1] = cand_len

    tokens, prop_spans = _place_props_sequential(rng, prop_ids, lengths, cfg.vocab_size)

    candidates = (0, 1)
    true_index = int(rng.integers(0, 2))
    true_prem = candidates[true_index]
    conclusion = n_props - 1

    edges: List[Tuple[int, int]] = []
    for i in range(cfg.n_mid):
        mid = cfg.n_cand_prem + i
        edges.append((true_prem, mid))
        edges.append((mid, conclusion))

    tokens = _append_edges(tokens, edges, cfg.T)

    return BackwardChainInstance(
        instance_id=instance_id,
        tokens=tokens,
        prop_spans=prop_spans,
        candidates=candidates,
        true_index=true_index,
    )


def _sample_dag_edges(
    rng: np.random.Generator,
    topo: Sequence[int],
    edge_prob: float,
    allow_nodes: Iterable[int],
) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    pos = {node: i for i, node in enumerate(topo)}
    allow = set(allow_nodes)
    for i, a in enumerate(topo):
        if a not in allow:
            continue
        for b in topo[i + 1 :]:
            if b not in allow:
                continue
            if rng.random() < edge_prob:
                edges.append((a, b))
    return edges


def _ensure_premise_outgoing(
    rng: np.random.Generator,
    topo: Sequence[int],
    premises: Sequence[int],
    edges: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    existing = {(a, b) for (a, b) in edges}
    pos = {node: i for i, node in enumerate(topo)}
    for prem in premises:
        outs = [b for (a, b) in existing if a == prem]
        if outs:
            continue
        candidates = [b for b in topo if pos[b] > pos[prem]]
        if not candidates:
            continue
        b = int(rng.choice(candidates))
        existing.add((prem, b))
    return list(existing)


def generate_cycle_regime(run_id: str, instance_id: str, cfg: CycleRegimeConfig) -> CycleRegimeInstance:
    """Generate a cycle-regime instance with exactly one k-cycle (k in {2,3,4}) or a DAG."""
    seed = instance_seed_u32(run_id, instance_id, salt="GEN_CYC_V1")
    rng = np.random.default_rng(seed)

    prop_ids = list(range(cfg.M))
    lengths = [int(rng.integers(cfg.prop_len_min, cfg.prop_len_max + 1)) for _ in prop_ids]
    tokens, prop_spans = _place_props_sequential(rng, prop_ids, lengths, cfg.vocab_size)

    premises = list(rng.choice(prop_ids, size=cfg.n_prem, replace=False))
    regime = int(rng.integers(0, 4))  # 0 DAG, 1 C2, 2 C3, 3 C4

    topo = list(rng.permutation(prop_ids))
    edges: List[Tuple[int, int]] = []

    cycle_nodes: List[int] = []
    if regime > 0:
        k = regime + 1
        cycle_nodes = list(rng.choice(prop_ids, size=k, replace=False))
        for i in range(k):
            edges.append((cycle_nodes[i], cycle_nodes[(i + 1) % k]))

    # Only add DAG edges among non-cycle nodes to avoid extra cycles.
    non_cycle = [n for n in prop_ids if n not in cycle_nodes]
    edges.extend(_sample_dag_edges(rng, topo, cfg.edge_prob, allow_nodes=non_cycle))
    edges = _ensure_premise_outgoing(rng, topo, premises, edges)

    tokens = _append_edges(tokens, edges, cfg.T)

    return CycleRegimeInstance(
        instance_id=instance_id,
        tokens=tokens,
        prop_spans=prop_spans,
        premises=premises,
        regime=regime,
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
