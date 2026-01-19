"""Deterministic per-instance seeding and matched-random sampling.

NHSEA-v1 preregistered rule:
- Per-instance RNG seed is the 32-bit unsigned integer given by the first 8 hex
  characters of SHA256(f"{run_id}:{instance_id}:{salt}") interpreted as uint32.

Matched-random sampling for token localization:
- Let R = [0..T-1] \ (A_prem \cup A_cand).
- If |R| >= |A_prem|, sample R0 subset of R of size |A_prem| without replacement.
- Else sample R0 subset of ([0..T-1] \ A_prem) of size |A_prem| without replacement.
- Sampling is deterministic using the hashed seed.

All indices are 0-based.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

import numpy as np


def instance_seed_u32(run_id: str, instance_id: str, salt: str = "RANDSEED_V1") -> int:
    """Return deterministic uint32 seed.

    The seed is defined as the first 8 hex chars of SHA256(f"{run_id}:{instance_id}:{salt}").
    """
    msg = f"{run_id}:{instance_id}:{salt}".encode("utf-8")
    h = hashlib.sha256(msg).hexdigest()
    first8 = h[:8]
    return int(first8, 16) & 0xFFFFFFFF


def _to_set_int(x: Iterable[int] | Set[int]) -> Set[int]:
    return set(int(i) for i in x)


def sample_R0(
    run_id: str,
    instance_id: str,
    T: int,
    A_prem: Iterable[int],
    A_cand: Iterable[int],
    salt: str = "RANDSEED_V1",
) -> List[int]:
    """Deterministically sample matched-random token indices R0.

    Returns a sorted list of indices (ascending), purely for determinism in downstream logs.
    """
    if T <= 0:
        raise ValueError("T must be positive")

    A_prem_s = _to_set_int(A_prem)
    A_cand_s = _to_set_int(A_cand)
    n = len(A_prem_s)
    if n == 0:
        return []

    universe = set(range(T))
    R = sorted(universe - (A_prem_s | A_cand_s))
    if len(R) >= n:
        pool = np.array(R, dtype=np.int64)
    else:
        pool = np.array(sorted(universe - A_prem_s), dtype=np.int64)
        if len(pool) < n:
            raise ValueError("Not enough tokens to sample without replacement")

    seed = instance_seed_u32(run_id, instance_id, salt=salt)
    rng = np.random.default_rng(seed)
    choice = rng.choice(pool, size=n, replace=False)
    return sorted(int(i) for i in choice)


@dataclass(frozen=True)
class RandSeedSpec:
    """Convenience container for debugging determinism."""

    run_id: str
    instance_id: str
    salt: str = "RANDSEED_V1"

    def seed(self) -> int:
        return instance_seed_u32(self.run_id, self.instance_id, self.salt)
