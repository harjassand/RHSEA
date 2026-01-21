"""NHSEA v3 generator with OBC vs PBC paired topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .seeding import instance_seed_u32


PAD_TOKEN = "PAD"
EDGE_TOKEN = "EDGE"
QRY_TOKEN = "QRY"
TASK_CONC_TOKEN = "TASK_CONC"
TASK_TOPO_TOKEN = "TASK_TOPO"


@dataclass(frozen=True)
class V3Config:
    T: int = 64
    M: int = 8
    K: int = 4
    L_min: int = 3
    L_max: int = 6
    vocab_size: int = 200


@dataclass(frozen=True)
class V3Instance:
    run_id: str
    instance_id: str
    pair_id: str
    topology: str
    tokens: List[str]
    spans: List[Tuple[int, int]]  # [start,end) per proposition span
    pi: List[int]
    premises: List[int]
    candidates: Tuple[int, int]
    true_index: int
    backward_true_index: int | None


def _prop_token(pid: int) -> str:
    return f"P{pid:02d}"


def _rand_word(rng: np.random.Generator, vocab_size: int) -> str:
    return f"w{int(rng.integers(0, vocab_size))}"


def _sample_lengths(rng: np.random.Generator, cfg: V3Config, edge_count: int) -> List[int]:
    # Ensure proposition spans fit within the fixed-length budget.
    max_prop_tokens = cfg.T - edge_count * 3 - 5
    min_total = cfg.M * cfg.L_min
    max_total = cfg.M * cfg.L_max
    if max_prop_tokens < min_total:
        raise ValueError("T too small for proposition spans with current config")
    capped_max = min(max_prop_tokens, max_total)
    total = int(rng.integers(min_total, capped_max + 1))
    lengths = [cfg.L_min for _ in range(cfg.M)]
    remaining = total - min_total
    caps = [cfg.L_max - cfg.L_min for _ in range(cfg.M)]
    while remaining > 0:
        idx = int(rng.integers(0, cfg.M))
        if caps[idx] > 0:
            lengths[idx] += 1
            caps[idx] -= 1
            remaining -= 1
    return lengths


def _build_props(
    rng: np.random.Generator,
    cfg: V3Config,
    edge_count: int,
) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    pi: List[int] = []

    lengths = _sample_lengths(rng, cfg, edge_count)
    for pid, length in enumerate(lengths):
        start = len(tokens)
        tokens.append(_prop_token(pid))
        for _ in range(length - 1):
            tokens.append(_rand_word(rng, cfg.vocab_size))
        end = len(tokens)
        spans.append((start, end))
        pi.extend([pid] * (end - start))

    return tokens, spans, pi


def _chain_nodes(rng: np.random.Generator, cfg: V3Config, premises: List[int]) -> List[int]:
    available = [i for i in range(cfg.M) if i not in premises]
    if len(available) < cfg.K + 1:
        raise ValueError("Not enough propositions to build chain")
    perm = rng.permutation(available)
    return [int(i) for i in perm[: cfg.K + 1]]


def _edges_for_topology(chain: List[int], premises: List[int], topology: str) -> List[Tuple[int, int]]:
    if len(chain) < 2:
        raise ValueError("Chain must have length >= 2")
    edges: List[Tuple[int, int]] = []
    for prem in premises:
        edges.append((prem, chain[0]))
    for i in range(len(chain) - 1):
        edges.append((chain[i], chain[i + 1]))

    if topology == "OBC":
        extra = (chain[-2], chain[-1])
    else:
        cycle_target = chain[1]
        extra = (chain[-1], cycle_target)
    edges.append(extra)
    return edges


def _has_cycle(n_nodes: int, edges: List[Tuple[int, int]]) -> bool:
    adj = {i: set() for i in range(n_nodes)}
    for src, dst in edges:
        adj[src].add(dst)
    state = [0] * n_nodes

    def _dfs(node: int) -> bool:
        state[node] = 1
        for nxt in adj.get(node, set()):
            if state[nxt] == 1:
                return True
            if state[nxt] == 0 and _dfs(nxt):
                return True
        state[node] = 2
        return False

    for n in range(n_nodes):
        if state[n] == 0 and _dfs(n):
            return True
    return False


def _check_graph(chain: List[int], premises: List[int], edges: List[Tuple[int, int]], topology: str) -> None:
    edge_set = set(edges)
    for prem in premises:
        if (prem, chain[0]) not in edge_set:
            raise AssertionError("Premise does not connect to chain start")
    for i in range(len(chain) - 1):
        if (chain[i], chain[i + 1]) not in edge_set:
            raise AssertionError("Missing chain edge")

    has_cycle = _has_cycle(max(chain + premises) + 1, edges)
    if topology == "OBC" and has_cycle:
        raise AssertionError("OBC graph must be acyclic")
    if topology == "PBC" and not has_cycle:
        raise AssertionError("PBC graph must contain a cycle")


def _append_edges(tokens: List[str], edges: List[Tuple[int, int]], T: int) -> List[str]:
    out = list(tokens)
    for src, dst in edges:
        out.extend([EDGE_TOKEN, _prop_token(src), _prop_token(dst)])
    if len(out) > T:
        raise ValueError("Token sequence exceeds T after adding edges")
    return out


def _build_pi(base_pi: List[int], total_len: int) -> List[int]:
    if len(base_pi) > total_len:
        raise ValueError("pi longer than token sequence")
    return base_pi + [-1] * (total_len - len(base_pi))


def generate_v3_instance(
    run_id: str,
    instance_id: str,
    pair_id: str,
    topology: str,
    task: str,
    cfg: V3Config,
) -> V3Instance:
    if topology not in ("OBC", "PBC"):
        raise ValueError("topology must be 'OBC' or 'PBC'")
    if task not in ("conclusion", "topology"):
        raise ValueError("task must be 'conclusion' or 'topology'")

    seed = instance_seed_u32(run_id, instance_id, salt="GEN_V3")
    rng = np.random.default_rng(seed)

    premises = [0, 1]
    edge_count = len(premises) + cfg.K + 1

    task_token = TASK_CONC_TOKEN if task == "conclusion" else TASK_TOPO_TOKEN
    tokens: List[str] = [task_token]
    spans_offset: List[Tuple[int, int]] = []
    pi: List[int] = [-1]

    prop_tokens, spans, prop_pi = _build_props(rng, cfg, edge_count)
    tokens.extend(prop_tokens)
    spans_offset = [(s + 1, e + 1) for (s, e) in spans]
    pi.extend(prop_pi)

    chain = _chain_nodes(rng, cfg, premises)
    edges = _edges_for_topology(chain, premises, topology)
    if __debug__:
        _check_graph(chain, premises, edges, topology)
    tokens = _append_edges(tokens, edges, cfg.T)

    # Query uses the first premise for consistency.
    query_prop = premises[0]
    tokens.extend([QRY_TOKEN, _prop_token(query_prop)])

    # Ensure space for candidates at fixed slots.
    if len(tokens) > cfg.T - 2:
        raise ValueError("Token sequence too long before candidates")

    pad_needed = cfg.T - 2 - len(tokens)
    if pad_needed > 0:
        tokens.extend([PAD_TOKEN] * pad_needed)

    conclusion = chain[-1]
    decoy_pool = [i for i in range(cfg.M) if i != conclusion and i not in (0, 1)]
    decoy = int(rng.choice(decoy_pool))

    if rng.random() < 0.5:
        candidates = (conclusion, decoy)
        true_index = 0
    else:
        candidates = (decoy, conclusion)
        true_index = 1

    tokens.extend([_prop_token(candidates[0]), _prop_token(candidates[1])])
    if len(tokens) != cfg.T:
        raise ValueError("Token sequence must equal T")

    full_pi = _build_pi(pi, len(tokens))

    return V3Instance(
        run_id=run_id,
        instance_id=instance_id,
        pair_id=pair_id,
        topology=topology,
        tokens=tokens,
        spans=spans_offset,
        pi=full_pi,
        premises=premises,
        candidates=candidates,
        true_index=true_index,
        backward_true_index=None,
    )
