import numpy as np

from nhsea.topk import rownorm_plus, diagzero, topkoffdiag_k, topk_rownorm


def test_rownorm_plus_all_zero_rows_stable():
    A = np.zeros((3, 3))
    P = rownorm_plus(A)
    assert np.all(P == 0)


def test_diagzero():
    A = np.ones((4, 4))
    B = diagzero(A)
    assert np.all(np.diag(B) == 0)
    assert np.all(B[np.arange(4)[:, None] != np.arange(4)[None, :]] == 1)


def test_topkoffdiag_k_tie_break_lowest_index():
    # Row 0: three equal positives, k=2 -> pick cols 1 and 2 (lowest indices)
    A = np.zeros((4, 4))
    A[0, 1] = 0.5
    A[0, 2] = 0.5
    A[0, 3] = 0.5
    B = topkoffdiag_k(A, k=2)
    assert B[0, 1] > 0 and B[0, 2] > 0 and B[0, 3] == 0


def test_topkoffdiag_k_keeps_all_if_fewer_than_k():
    A = np.zeros((4, 4))
    A[1, 0] = 0.2
    B = topkoffdiag_k(A, k=3)
    assert B[1, 0] == 0.2
    assert np.count_nonzero(B[1]) == 1


def test_topk_rownorm_rowsum_one_when_nonzero():
    A = np.zeros((3, 3))
    A[0, 1] = 2.0
    A[0, 2] = 1.0
    P = topk_rownorm(A, k=2)
    assert np.isclose(P[0].sum(), 1.0)
    assert np.all(P[1] == 0)
