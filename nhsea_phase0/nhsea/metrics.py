"""Preregged Phase 0 metrics: token SelLocGap, proposition cycles, and PR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .operator import OperatorSpec, build_run_operator, token_weights
from .seeding import instance_seed_u32, sample_R0
from .topk import diagzero, rownorm_plus, topk_rownorm


@dataclass(frozen=True)
class SelLocResult:
    adj_locgap_mech: float
    adj_locgap_sym: float
    sel_loc_gap: float


def _seed_vector(indices: Sequence[int], size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float64)
    if not indices:
        return v
    scale = 1.0 / np.sqrt(len(indices))
    v[np.asarray(indices, dtype=np.int64)] = scale
    return v


def _propagate_normalize(W: np.ndarray, v0: np.ndarray, steps: int) -> np.ndarray:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if steps == 0:
        v = v0.astype(np.float64)
    else:
        v = np.linalg.matrix_power(W, steps) @ v0
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return np.zeros_like(v, dtype=np.float64)
    return v / norm


def _loc_gap(v: np.ndarray, cand_true: Sequence[int], cand_false: Sequence[int]) -> float:
    true_idx = np.asarray(cand_true, dtype=np.int64)
    false_idx = np.asarray(cand_false, dtype=np.int64)
    loc_true = float(np.sum(v[true_idx] ** 2))
    loc_false = float(np.sum(v[false_idx] ** 2))
    return loc_true - loc_false


def token_topk_rownorm(
    L: np.ndarray,
    U: np.ndarray,
    spec: OperatorSpec,
    k_tok: int,
) -> np.ndarray:
    """Build W_tok_hat = TopK+RowNorm(W_tok) for a variant."""
    O = build_run_operator(L, U, spec)
    W_tok = token_weights(O)
    return topk_rownorm(W_tok, k_tok)


def token_sel_loc_gap(
    L: np.ndarray,
    U: np.ndarray,
    spec_mech: OperatorSpec,
    spec_sym: OperatorSpec,
    k_tok: int,
    prem_tokens: Sequence[int],
    cand_true_tokens: Sequence[int],
    cand_false_tokens: Sequence[int],
    run_id: str,
    instance_id: str,
) -> SelLocResult:
    """Compute SelLocGap with prem/rand adjustment for mech vs symmetric control."""
    T = L.shape[0]
    if T <= 0:
        raise ValueError("T must be positive")
    if L.shape[0] != L.shape[1] or U.shape != L.shape:
        raise ValueError("L and U must be square and same shape")
    if spec_mech.variant != "mechanism":
        raise ValueError("spec_mech.variant must be 'mechanism'")
    if spec_sym.variant != "symmetric_control":
        raise ValueError("spec_sym.variant must be 'symmetric_control'")

    s_tok = min(5, T)
    cand_all = list(cand_true_tokens) + list(cand_false_tokens)
    rand_tokens = sample_R0(run_id, instance_id, T, prem_tokens, cand_all)

    v0_prem = _seed_vector(prem_tokens, T)
    v0_rand = _seed_vector(rand_tokens, T)

    def adj_locgap_for(spec: OperatorSpec) -> float:
        W_hat = token_topk_rownorm(L, U, spec, k_tok)
        v_prem = _propagate_normalize(W_hat, v0_prem, s_tok)
        v_rand = _propagate_normalize(W_hat, v0_rand, s_tok)
        locgap_prem = _loc_gap(v_prem, cand_true_tokens, cand_false_tokens)
        locgap_rand = _loc_gap(v_rand, cand_true_tokens, cand_false_tokens)
        return locgap_prem - locgap_rand

    adj_mech = adj_locgap_for(spec_mech)
    adj_sym = adj_locgap_for(spec_sym)
    return SelLocResult(adj_locgap_mech=adj_mech, adj_locgap_sym=adj_sym, sel_loc_gap=adj_mech - adj_sym)


def _prop_token_sets(prop_spans: Sequence[Tuple[int, int]]) -> List[np.ndarray]:
    sets: List[np.ndarray] = []
    for (s, e) in prop_spans:
        if e < s:
            raise ValueError("Invalid span")
        sets.append(np.arange(s, e + 1, dtype=np.int64))
    return sets


def _collapse_to_prop(W_tok: np.ndarray, prop_spans: Sequence[Tuple[int, int]]) -> np.ndarray:
    sets = _prop_token_sets(prop_spans)
    M = len(sets)
    W_prop = np.zeros((M, M), dtype=np.float64)
    for a, idx_a in enumerate(sets):
        for b, idx_b in enumerate(sets):
            sub = W_tok[np.ix_(idx_a, idx_b)]
            if sub.size == 0:
                W_prop[a, b] = 0.0
            else:
                W_prop[a, b] = float(sub.mean())
    return W_prop


def prop_pipeline_a(
    W_tok: np.ndarray,
    prop_spans: Sequence[Tuple[int, int]],
    k_tok: int,
    k_prop: int,
) -> np.ndarray:
    """Pipeline A: token TopK+RowNorm -> collapse -> prop TopK+RowNorm -> DiagZero -> RowNorm+."""
    W_tok_hat = topk_rownorm(W_tok, k_tok)
    W_prop = _collapse_to_prop(W_tok_hat, prop_spans)
    W_prop_bar = topk_rownorm(W_prop, k_prop)
    return rownorm_plus(diagzero(W_prop_bar))


def prop_pipeline_b(
    W_tok: np.ndarray,
    prop_spans: Sequence[Tuple[int, int]],
    k_prop: int,
) -> np.ndarray:
    """Pipeline B: collapse -> prop TopK+RowNorm -> DiagZero -> RowNorm+."""
    W_prop = _collapse_to_prop(W_tok, prop_spans)
    W_prop_bar = topk_rownorm(W_prop, k_prop)
    return rownorm_plus(diagzero(W_prop_bar))


def cyc_stats(W_prop_hat: np.ndarray, ells: Sequence[int] = (2, 3, 4)) -> dict[int, float]:
    out: dict[int, float] = {}
    for ell in ells:
        if ell < 2:
            raise ValueError("ell must be >= 2")
        out[ell] = float(np.trace(np.linalg.matrix_power(W_prop_hat, ell)) / ell)
    return out


def permrow_null(
    W_prop_hat: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Row-preserving permutation null on fixed off-diagonal support."""
    rng = np.random.default_rng(seed)
    W = np.asarray(W_prop_hat, dtype=np.float64)
    n = W.shape[0]
    out = np.zeros_like(W, dtype=np.float64)
    for a in range(n):
        row = W[a]
        idx = [b for b in range(n) if b != a and row[b] > 0]
        if not idx:
            continue
        vals = np.array([row[b] for b in idx], dtype=np.float64)
        if len(vals) > 1:
            vals = vals[rng.permutation(len(vals))]
        for b, v in zip(idx, vals):
            out[a, b] = v
    return rownorm_plus(diagzero(out))


def delta_cyc_stats(
    W_prop_hat: np.ndarray,
    run_id: str,
    instance_id: str,
    n_perm: int = 100,
    ells: Sequence[int] = (2, 3, 4),
    salt_prefix: str = "PERMROW",
) -> dict[int, float]:
    base = cyc_stats(W_prop_hat, ells=ells)
    accum = {ell: 0.0 for ell in ells}
    for r in range(n_perm):
        seed = instance_seed_u32(run_id, instance_id, salt=f"{salt_prefix}_{r}")
        W_null = permrow_null(W_prop_hat, seed)
        null_stats = cyc_stats(W_null, ells=ells)
        for ell in ells:
            accum[ell] += null_stats[ell]
    for ell in ells:
        accum[ell] /= float(n_perm)
    return {ell: base[ell] - accum[ell] for ell in ells}


def participation_ratio(
    W_prop_hat: np.ndarray,
    prem_prop_ids: Iterable[int],
) -> float:
    W = np.asarray(W_prop_hat, dtype=np.float64)
    M = W.shape[0]
    prem = list(int(i) for i in prem_prop_ids)
    if M == 0 or not prem:
        return 0.0
    u0 = _seed_vector(prem, M)
    s = min(5, M)
    u = _propagate_normalize(W, u0, s)
    denom = float(np.sum(np.abs(u) ** 4))
    if denom == 0.0:
        return 0.0
    return 1.0 / denom
