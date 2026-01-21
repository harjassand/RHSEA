import numpy as np

from analysis_asymmetry_report import _asym_ratio


def test_asym_ratio_symmetric_matrix_near_zero():
    base = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float64)
    assert _asym_ratio(base) < 1e-12


def test_asym_ratio_random_matrix_positive():
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(5, 5))
    assert _asym_ratio(mat) > 1e-6
