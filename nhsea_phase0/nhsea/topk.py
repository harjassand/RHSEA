"""NHSEA Phase 0 deterministic sparsification operators."""

from __future__ import annotations

import numpy as np


def rownorm_plus(A: np.ndarray) -> np.ndarray:
    """Row-normalize nonnegative A; all-zero rows remain all-zero."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {A.shape}")
    if np.any(A < 0):
        raise ValueError("RowNorm^+ expects a nonnegative matrix")

    row_sums = A.sum(axis=1)
    out = np.zeros_like(A, dtype=np.float64)
    nz = row_sums > 0
    out[nz, :] = A[nz, :] / row_sums[nz, None]
    return out


def diagzero(A: np.ndarray) -> np.ndarray:
    """Set diagonal to zero (returns a copy)."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {A.shape}")
    out = A.copy()
    n = min(out.shape[0], out.shape[1])
    out[np.arange(n), np.arange(n)] = 0.0
    return out


def topkoffdiag_k(A: np.ndarray, k: int) -> np.ndarray:
    """Keep up to k largest strictly-positive off-diagonal entries per row.

    Tie-break: lowest column index.
    If fewer than k strictly-positive off-diagonal entries exist in a row, keep all.
    Diagonal is always set to 0.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {A.shape}")
    n, m = A.shape
    if n != m:
        raise ValueError(f"Expected square matrix, got {A.shape}")
    if not (1 <= k <= n - 1):
        raise ValueError(f"k must be in [1, n-1], got k={k}, n={n}")
    if np.any(A < 0):
        raise ValueError("TopKOffDiag expects a nonnegative matrix")

    B = np.zeros_like(A, dtype=np.float64)
    for a in range(n):
        row = A[a, :]
        eligible = [b for b in range(n) if b != a and row[b] > 0]
        if not eligible:
            continue
        # Sort by value descending, then column ascending.
        eligible_sorted = sorted(eligible, key=lambda b: (-row[b], b))
        keep = eligible_sorted[:k] if len(eligible_sorted) >= k else eligible_sorted
        for b in keep:
            B[a, b] = row[b]
        # Ensure diagonal is zero.
        B[a, a] = 0.0
    return B


def topk_rownorm(A: np.ndarray, k: int) -> np.ndarray:
    """TopKOffDiag_k(A) then RowNorm^+."""
    return rownorm_plus(topkoffdiag_k(A, k))
