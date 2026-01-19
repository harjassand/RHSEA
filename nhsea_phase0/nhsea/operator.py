"""NHSEA Phase 0 operator construction: extraction, drift, skew, clamp, weights.

This code follows the preregistered definitions:
- Extract attention logits at fixed layer by averaging heads (provided externally).
- Drift matrix D_{ij}=(j-i)/T.
- Skew S = U - U^T, dummy symmetric S_dummy = U + U^T.
- O_raw = L + alpha D + beta S (or control variants).
- Clamp maps NaN and +/-inf to z_min, clips otherwise to [z_min,z_max].
- sigma is logistic; token diagonal zeroed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


def extract_logits(attn_logits_layer: np.ndarray) -> np.ndarray:
    """Average heads at the fixed probe layer.

    Expects shape (H, T, T) and returns (T, T).
    """
    arr = np.asarray(attn_logits_layer, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected (H,T,T) array, got {arr.shape}")
    return arr.mean(axis=0)


def drift_matrix(T: int) -> np.ndarray:
    """Compute D_{ij}=(j-i)/T."""
    if T <= 0:
        raise ValueError("T must be positive")
    i = np.arange(T, dtype=np.float64)[:, None]
    j = np.arange(T, dtype=np.float64)[None, :]
    return (j - i) / float(T)


def clamp_finite(A: np.ndarray, z_min: float = -30.0, z_max: float = 30.0, **kwargs) -> np.ndarray:
    """Deterministic unified clamp.

    Maps NaN, -inf, +inf to z_min. Clips finite values to [z_min, z_max].
    """
    # Support alias keyword arguments used by some call sites (zmin/zmax).
    if 'zmin' in kwargs:
        z_min = float(kwargs['zmin'])
    if 'zmax' in kwargs:
        z_max = float(kwargs['zmax'])

    A = np.asarray(A, dtype=np.float64)
    out = A.copy()
    # Map non-finite to z_min.
    bad = ~np.isfinite(out)
    out[bad] = z_min
    # Clip remaining values.
    np.clip(out, z_min, z_max, out=out)
    return out


def sigma_logistic(Z: np.ndarray) -> np.ndarray:
    """Elementwise logistic sigma(z) = 1/(1+exp(-z))."""
    Z = np.asarray(Z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-Z))


Variant = Literal["mechanism", "symmetric_control", "no_injection", "no_drift"]


@dataclass(frozen=True)
class OperatorSpec:
    alpha: float
    beta: float
    variant: Variant
    z_min: float = -30.0
    z_max: float = 30.0


def build_run_operator(
    L: np.ndarray,
    U: Optional[np.ndarray],
    spec: OperatorSpec,
) -> np.ndarray:
    """Build O_theta(x) from extracted logits L and auxiliary pathway U.

    - L: (T,T) attention logits (already extracted / averaged over heads).
    - U: (T,T) auxiliary pathway output. Required for variants using injection.
    """
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square (T,T); got {L.shape}")
    T = L.shape[0]

    D = drift_matrix(T)

    if spec.variant in ("mechanism", "symmetric_control", "no_drift"):
        if U is None:
            raise ValueError("U must be provided for injection variants")
        U = np.asarray(U, dtype=np.float64)
        if U.shape != (T, T):
            raise ValueError(f"U must have shape (T,T)={L.shape}, got {U.shape}")

    if spec.variant == "mechanism":
        S = U - U.T
        O_raw = L + spec.alpha * D + spec.beta * S
    elif spec.variant == "symmetric_control":
        S_dummy = U + U.T
        O_raw = L + spec.alpha * D + spec.beta * S_dummy
    elif spec.variant == "no_injection":
        O_raw = L + spec.alpha * D
    elif spec.variant == "no_drift":
        S = U - U.T
        O_raw = L + spec.beta * S
    else:
        raise ValueError(f"Unknown variant: {spec.variant}")

    return clamp_finite(O_raw, z_min=spec.z_min, z_max=spec.z_max)


def token_weights(O: np.ndarray) -> np.ndarray:
    """Compute token-level weights W_tok = sigma(O) and zero diagonal."""
    W = sigma_logistic(O)
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"Expected square matrix, got {W.shape}")
    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 0.0
    return W


def scale_norms(U: np.ndarray, beta: float) -> Tuple[float, float]:
    """Compute A=||beta S||_F and B=||beta S_dummy||_F for mismatch logging."""
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"U must be square, got {U.shape}")
    S = U - U.T
    S_dummy = U + U.T
    A = float(np.linalg.norm(beta * S, ord="fro"))
    B = float(np.linalg.norm(beta * S_dummy, ord="fro"))
    return A, B


def rho_ratio(A: float, B: float) -> float:
    """Compute rho as in prereg spec."""
    if A == 0.0 and B == 0.0:
        return 1.0
    if A > 0.0 and B == 0.0:
        return float("inf")
    if A == 0.0 and B > 0.0:
        return 0.0
    return A / B

# Aliases used by scripts/tests
sigmoid = sigma_logistic
weights_from_operator = token_weights
