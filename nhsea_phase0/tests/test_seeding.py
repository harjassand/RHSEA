import numpy as np

from nhsea.seeding import instance_seed_u32, sample_R0


def test_instance_seed_u32_deterministic():
    s1 = instance_seed_u32("runA", "inst1")
    s2 = instance_seed_u32("runA", "inst1")
    assert s1 == s2
    s3 = instance_seed_u32("runA", "inst2")
    assert s3 != s1


def test_sample_R0_size_and_exclusion():
    run_id = "runA"
    instance_id = "inst1"
    T = 50
    A_prem = list(range(10))
    A_cand = list(range(10, 20))
    R0 = sample_R0(run_id, instance_id, T, A_prem, A_cand)
    assert len(R0) == len(A_prem)
    # If pool sufficient, should avoid A_prem and A_cand
    assert set(R0).isdisjoint(set(A_prem))
    assert set(R0).isdisjoint(set(A_cand))


def test_sample_R0_fallback_when_pool_too_small():
    run_id = "runA"
    instance_id = "instX"
    T = 12
    A_prem = list(range(6))
    A_cand = list(range(6, 12))  # leaves empty R
    R0 = sample_R0(run_id, instance_id, T, A_prem, A_cand)
    assert len(R0) == len(A_prem)
    assert set(R0).isdisjoint(set(A_prem))
