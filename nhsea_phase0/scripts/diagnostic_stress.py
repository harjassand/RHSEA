#!/usr/bin/env python
"""Diagnostic stress tests for prereg metrics (no SGD).

Layer A: pure-math golden tests on proposition-level W_hat (row-stochastic, diag zero).
Layer B: pipeline-integrated tests from token weights through TopK/collapse/PermRow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from nhsea.metrics import cyc_stats, delta_cyc_stats, participation_ratio, prop_pipeline_a, prop_pipeline_b
from nhsea.seeding import sample_R0
from nhsea.topk import diagzero, rownorm_plus, topk_rownorm


def sigma_logistic(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-Z))


def build_W_tok_from_O(O: np.ndarray) -> np.ndarray:
    W = sigma_logistic(O)
    return diagzero(W)


def make_block_spans(T: int, M: int) -> List[Tuple[int, int]]:
    if T % M != 0:
        raise ValueError("T must be divisible by M for equal blocks")
    block = T // M
    return [(i * block, i * block + block - 1) for i in range(M)]


def span_tokens(span: Tuple[int, int]) -> List[int]:
    s, e = span
    return list(range(s, e + 1))


def seed_vector(indices: Sequence[int], size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float64)
    if not indices:
        return v
    v[indices] = 1.0 / np.sqrt(len(indices))
    return v


def propagate_normalize(W: np.ndarray, v0: np.ndarray, steps: int) -> np.ndarray:
    if steps == 0:
        v = v0.astype(np.float64)
    else:
        v = np.linalg.matrix_power(W, steps) @ v0
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return np.zeros_like(v)
    return v / norm


def loc_gap(v: np.ndarray, cand_true: Sequence[int], cand_false: Sequence[int]) -> float:
    loc_true = float(np.sum(v[np.asarray(cand_true, dtype=np.int64)] ** 2))
    loc_false = float(np.sum(v[np.asarray(cand_false, dtype=np.int64)] ** 2))
    return loc_true - loc_false


def sel_loc_gap_from_O(
    O_mech: np.ndarray,
    O_sym: np.ndarray,
    prem_tokens: Sequence[int],
    cand_true_tokens: Sequence[int],
    cand_false_tokens: Sequence[int],
    run_id: str,
    instance_id: str,
    k_tok: int,
) -> Tuple[float, Dict[str, int]]:
    T = O_mech.shape[0]
    s_tok = min(5, T)
    cand_all = list(cand_true_tokens) + list(cand_false_tokens)
    rand_tokens = sample_R0(run_id, instance_id, T, prem_tokens, cand_all)
    v0_prem = seed_vector(prem_tokens, T)
    v0_rand = seed_vector(rand_tokens, T)

    R = [i for i in range(T) if i not in set(prem_tokens) and i not in set(cand_all)]
    info = {
        "prem_size": len(prem_tokens),
        "cand_size": len(cand_all),
        "R_size": len(R),
        "fallback": int(len(R) < len(prem_tokens)),
    }

    def adj_locgap(O: np.ndarray) -> float:
        W_tok = build_W_tok_from_O(O)
        W_hat = topk_rownorm(W_tok, k_tok)
        v_prem = propagate_normalize(W_hat, v0_prem, s_tok)
        v_rand = propagate_normalize(W_hat, v0_rand, s_tok)
        return loc_gap(v_prem, cand_true_tokens, cand_false_tokens) - loc_gap(v_rand, cand_true_tokens, cand_false_tokens)

    return adj_locgap(O_mech) - adj_locgap(O_sym), info


def make_chain_blocks(T: int, spans: List[Tuple[int, int]], a: float, bidirectional: bool = False) -> np.ndarray:
    O = np.zeros((T, T), dtype=np.float64)
    for i in range(len(spans) - 1):
        src = span_tokens(spans[i])
        dst = span_tokens(spans[i + 1])
        for s in src:
            for d in dst:
                O[s, d] = a
                if bidirectional:
                    O[d, s] = a
    return O


def make_antisymmetric(T: int, b: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(T, T))
    S = R - R.T
    Sym = R + R.T
    return b * S, b * Sym


def build_prop_dag(M: int, w_hi: float, w_lo: float) -> np.ndarray:
    """DAG with matched out-degree (2 for nodes 0..5, 0 for nodes 6..7)."""
    W = np.zeros((M, M), dtype=np.float64)
    sink_a, sink_b = M - 2, M - 1
    for i in range(M - 2):
        hi = i + 1
        lo = sink_a if i % 2 == 0 else sink_b
        if hi == lo:
            lo = sink_b if lo == sink_a else sink_a
        W[i, hi] = w_hi
        W[i, lo] = w_lo
    return rownorm_plus(diagzero(W))


def build_prop_cycle(M: int, k: int, w_hi: float, w_lo: float, sym: bool = False) -> np.ndarray:
    """Directed k-cycle embedded in a DAG backbone; low edges point to sinks."""
    if k not in (2, 3, 4):
        raise ValueError("k must be 2, 3, or 4")
    W = build_prop_dag(M, w_hi, w_lo)
    sink_a, sink_b = M - 2, M - 1
    cycle_nodes = list(range(2, 2 + k))
    for i, node in enumerate(cycle_nodes):
        nxt = cycle_nodes[(i + 1) % k]
        prev = cycle_nodes[(i - 1) % k]
        W[node, :] = 0.0
        if sym and k > 2:
            W[node, nxt] = w_hi
            W[node, prev] = w_lo
        else:
            W[node, nxt] = w_hi
            W[node, sink_a if node % 2 == 0 else sink_b] = w_lo
    return rownorm_plus(diagzero(W))


def expand_prop_to_token(W_prop: np.ndarray, spans: List[Tuple[int, int]]) -> np.ndarray:
    T = spans[-1][1] + 1
    W_tok = np.zeros((T, T), dtype=np.float64)
    for a, span_a in enumerate(spans):
        src = span_tokens(span_a)
        for b, span_b in enumerate(spans):
            w = W_prop[a, b]
            if w <= 0:
                continue
            dst = span_tokens(span_b)
            for s in src:
                for d in dst:
                    W_tok[s, d] = w
    return W_tok


def golden_layer(T: int, M: int, w_hi: float, w_lo: float, n_perm: int) -> Tuple[Dict, Dict]:
    results: Dict[str, Dict] = {}
    criteria: Dict[str, bool] = {}

    W_dag = build_prop_dag(M, w_hi, w_lo)
    results["DAG"] = {
        "Cyc": {str(k): float(v) for k, v in cyc_stats(W_dag).items()},
        "DeltaCyc": {str(k): float(v) for k, v in delta_cyc_stats(W_dag, f"gold_T{T}", "DAG", n_perm).items()},
        "PR": participation_ratio(W_dag, [0, 1]),
    }

    for k in (2, 3, 4):
        W_dir = build_prop_cycle(M, k, w_hi, w_lo, sym=False)
        W_sym = build_prop_cycle(M, k, w_hi, w_lo, sym=True)
        key = f"C{k}"
        results[key] = {
            "Cyc": {str(j): float(v) for j, v in cyc_stats(W_dir).items()},
            "DeltaCyc": {str(j): float(v) for j, v in delta_cyc_stats(W_dir, f"gold_T{T}", key, n_perm).items()},
            "DeltaCyc_sym": {str(j): float(v) for j, v in delta_cyc_stats(W_sym, f"gold_T{T}", f"{key}_sym", n_perm).items()},
            "PR": participation_ratio(W_dir, [0, 1]),
        }

        cyc = results[key]["Cyc"]
        peak = max((2, 3, 4), key=lambda j: cyc[str(j)])
        criteria[f"gold_cyc_peak_T{T}_C{k}"] = bool(peak == k)
        dag_delta = results["DAG"]["DeltaCyc"][str(k)]
        dir_delta = results[key]["DeltaCyc"][str(k)]
        sym_delta = results[key]["DeltaCyc_sym"][str(k)]
        criteria[f"gold_delta_pos_T{T}_C{k}"] = bool((dir_delta > dag_delta) and (dir_delta > sym_delta))

    pr_dag = results["DAG"]["PR"]
    pr_cycles = np.mean([results[f"C{k}"]["PR"] for k in (2, 3, 4)])
    criteria[f"gold_pr_spread_T{T}"] = bool(pr_cycles > pr_dag)
    return results, criteria


def integrated_layer(T: int, M: int, w_hi: float, w_lo: float, n_perm: int) -> Tuple[Dict, Dict]:
    results: Dict[str, Dict] = {}
    criteria: Dict[str, bool] = {}

    spans = make_block_spans(T, M)
    block = T // M
    k_tok = 2 * block
    k_prop = 2

    # Chain localization test.
    prem_props = [0, 1]
    cand_true_prop = 6
    cand_false_prop = 7
    prem_tokens = span_tokens(spans[prem_props[0]]) + span_tokens(spans[prem_props[1]])
    cand_true_tokens = span_tokens(spans[cand_true_prop])
    cand_false_tokens = span_tokens(spans[cand_false_prop])

    O_chain = make_chain_blocks(T, spans, a=5.0, bidirectional=False)
    O_chain_sym = make_chain_blocks(T, spans, a=5.0, bidirectional=True)
    sel, info = sel_loc_gap_from_O(
        O_chain,
        O_chain_sym,
        prem_tokens,
        cand_true_tokens,
        cand_false_tokens,
        run_id=f"stress_T{T}",
        instance_id=f"chain_T{T}",
        k_tok=k_tok,
    )
    results["chain"] = {"SelLocGap": sel, "seed_info": info}
    criteria[f"chain_localization_T{T}"] = bool(sel > 0.0)

    # Cycle diagnostics through token pipeline.
    W_dag = build_prop_dag(M, w_hi, w_lo)
    W_tok_dag = expand_prop_to_token(W_dag, spans)
    W_prop_a_dag = prop_pipeline_a(W_tok_dag, spans, k_tok=k_tok, k_prop=k_prop)
    W_prop_b_dag = prop_pipeline_b(W_tok_dag, spans, k_prop=k_prop)
    dag_delta_a = delta_cyc_stats(W_prop_a_dag, f"pipe_T{T}", "DAG_A", n_perm=n_perm, ells=(2, 3, 4), salt_prefix="PERMROW_A")
    dag_delta_b = delta_cyc_stats(W_prop_b_dag, f"pipe_T{T}", "DAG_B", n_perm=n_perm, ells=(2, 3, 4), salt_prefix="PERMROW_B")

    for k in (2, 3, 4):
        W_dir = build_prop_cycle(M, k, w_hi, w_lo, sym=False)
        W_sym = build_prop_cycle(M, k, w_hi, w_lo, sym=True)
        W_tok_dir = expand_prop_to_token(W_dir, spans)
        W_tok_sym = expand_prop_to_token(W_sym, spans)

        W_prop_a = prop_pipeline_a(W_tok_dir, spans, k_tok=k_tok, k_prop=k_prop)
        W_prop_b = prop_pipeline_b(W_tok_dir, spans, k_prop=k_prop)
        W_prop_a_sym = prop_pipeline_a(W_tok_sym, spans, k_tok=k_tok, k_prop=k_prop)
        W_prop_b_sym = prop_pipeline_b(W_tok_sym, spans, k_prop=k_prop)

        delta_a = delta_cyc_stats(W_prop_a, f"pipe_T{T}", f"C{k}_A", n_perm=n_perm, ells=(2, 3, 4), salt_prefix="PERMROW_A")
        delta_b = delta_cyc_stats(W_prop_b, f"pipe_T{T}", f"C{k}_B", n_perm=n_perm, ells=(2, 3, 4), salt_prefix="PERMROW_B")
        delta_a_sym = delta_cyc_stats(W_prop_a_sym, f"pipe_T{T}", f"C{k}_A_sym", n_perm=n_perm, ells=(2, 3, 4), salt_prefix="PERMROW_A")
        delta_b_sym = delta_cyc_stats(W_prop_b_sym, f"pipe_T{T}", f"C{k}_B_sym", n_perm=n_perm, ells=(2, 3, 4), salt_prefix="PERMROW_B")

        results[f"C{k}"] = {
            "DeltaCyc_A": {str(j): float(v) for j, v in delta_a.items()},
            "DeltaCyc_B": {str(j): float(v) for j, v in delta_b.items()},
            "DeltaCyc_A_sym": {str(j): float(v) for j, v in delta_a_sym.items()},
            "DeltaCyc_B_sym": {str(j): float(v) for j, v in delta_b_sym.items()},
            "PR_A": participation_ratio(W_prop_a, prem_props),
            "PR_B": participation_ratio(W_prop_b, prem_props),
        }

        # Peak at matching ell.
        peak_a = max((2, 3, 4), key=lambda j: delta_a[j])
        peak_b = max((2, 3, 4), key=lambda j: delta_b[j])
        criteria[f"pipe_peak_A_T{T}_C{k}"] = bool(peak_a == k)
        criteria[f"pipe_peak_B_T{T}_C{k}"] = bool(peak_b == k)

        # Positive separation vs DAG and sym control (skip sym compare for k=2).
        criteria[f"pipe_delta_A_T{T}_C{k}"] = bool((delta_a[k] > dag_delta_a[k]) and (k == 2 or delta_a[k] > delta_a_sym[k]))
        criteria[f"pipe_delta_B_T{T}_C{k}"] = bool((delta_b[k] > dag_delta_b[k]) and (k == 2 or delta_b[k] > delta_b_sym[k]))

    # PR spread check: cycles pooled vs DAG.
    pr_dag_a = participation_ratio(W_prop_a_dag, prem_props)
    pr_dag_b = participation_ratio(W_prop_b_dag, prem_props)
    pr_cycles_a = np.mean([results[f"C{k}"]["PR_A"] for k in (2, 3, 4)])
    pr_cycles_b = np.mean([results[f"C{k}"]["PR_B"] for k in (2, 3, 4)])
    criteria[f"pipe_pr_A_T{T}"] = bool(pr_cycles_a > pr_dag_a)
    criteria[f"pipe_pr_B_T{T}"] = bool(pr_cycles_b > pr_dag_b)

    # Antisymmetric diagnostic case (report-only).
    O_anti, O_sym = make_antisymmetric(T, b=5.0, seed=0)
    W_tok_anti = build_W_tok_from_O(O_anti)
    W_prop_a_anti = prop_pipeline_a(W_tok_anti, spans, k_tok=k_tok, k_prop=k_prop)
    W_prop_b_anti = prop_pipeline_b(W_tok_anti, spans, k_prop=k_prop)
    results["antisym"] = {
        "DeltaCyc_A": {str(j): float(v) for j, v in delta_cyc_stats(W_prop_a_anti, f"pipe_T{T}", "antisym_A", n_perm=n_perm).items()},
        "DeltaCyc_B": {str(j): float(v) for j, v in delta_cyc_stats(W_prop_b_anti, f"pipe_T{T}", "antisym_B", n_perm=n_perm).items()},
        "PR_A": participation_ratio(W_prop_a_anti, prem_props),
        "PR_B": participation_ratio(W_prop_b_anti, prem_props),
    }

    results["k_tok"] = k_tok
    results["k_prop"] = k_prop
    return results, criteria


def main() -> int:
    w_hi, w_lo = 0.9, 0.1
    n_perm = 100

    report: Dict[str, Dict] = {"golden": {}, "pipeline": {}, "criteria": {}}
    for T in (32, 64):
        M = 8
        gold_res, gold_crit = golden_layer(T, M, w_hi, w_lo, n_perm)
        pipe_res, pipe_crit = integrated_layer(T, M, w_hi, w_lo, n_perm)
        report["golden"][f"T{T}"] = gold_res
        report["pipeline"][f"T{T}"] = pipe_res
        report["criteria"].update(gold_crit)
        report["criteria"].update(pipe_crit)

    report["passed"] = all(report["criteria"].values())

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "diagnostic_stress_report.json"
    txt_path = out_dir / "diagnostic_stress_report.txt"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    lines: List[str] = ["Stress diagnostics summary", f"passed={report['passed']}"]
    for key, val in sorted(report["criteria"].items()):
        lines.append(f"{key}: {'PASS' if val else 'FAIL'}")
    txt_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
