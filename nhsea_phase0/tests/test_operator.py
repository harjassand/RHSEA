import numpy as np

from nhsea.operator import clamp_finite, drift_matrix, sigmoid, weights_from_operator


def test_clamp_maps_sentinels_to_zmin_and_clips():
    zmin, zmax = -30.0, 30.0
    A = np.array([[np.nan, np.inf, -np.inf, 40.0, -40.0]], dtype=np.float64)
    B = clamp_finite(A, zmin=zmin, zmax=zmax)
    assert B[0, 0] == zmin
    assert B[0, 1] == zmin
    assert B[0, 2] == zmin
    assert B[0, 3] == zmax
    assert B[0, 4] == zmin


def test_drift_matrix_range_and_diagonal_zero():
    D = drift_matrix(5)
    assert D.shape == (5, 5)
    assert np.all(D <= 1.0 + 1e-12)
    assert np.all(D >= -1.0 - 1e-12)
    assert np.allclose(np.diag(D), 0.0)


def test_weights_from_operator_sigmoid_and_diag_zero():
    O = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    W = weights_from_operator(O)
    # sigmoid(0)=0.5 off-diagonal, diagonal forced 0
    assert W[0, 0] == 0.0 and W[1, 1] == 0.0
    assert np.isclose(W[0, 1], 0.5) and np.isclose(W[1, 0], 0.5)
