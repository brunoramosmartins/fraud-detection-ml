import numpy as np

from src.utils.drift import compute_psi


def test_compute_psi_identical_distributions():
    """Identical distributions must produce PSI ≈ 0."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 500)
    cur = ref.copy()
    psi = compute_psi(ref, cur, n_bins=10)
    assert 0.0 <= psi < 0.05


def test_compute_psi_different_distributions_exceeds_warning_threshold():
    """
    Clearly shifted distributions must produce PSI > 0.2 (the standard
    'significant drift' threshold used in monitor_model.py).
    """
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(5, 1, 500)  # Mean shifted by 5σ — unambiguous drift
    psi = compute_psi(ref, cur, n_bins=10)
    assert psi > 0.2


def test_compute_psi_returns_nonneg_for_skewed_distributions():
    """PSI must always be non-negative regardless of shape differences."""
    rng = np.random.default_rng(0)
    ref = rng.exponential(scale=1.0, size=200)
    cur = rng.exponential(scale=4.0, size=200)
    psi = compute_psi(ref, cur)
    assert psi >= 0.0


def test_compute_psi_constant_reference_does_not_raise():
    """A constant reference array triggers the fallback bin-edge path; must not raise."""
    ref = np.ones(50)
    cur = np.array([1.0, 2.0, 3.0] * 10 + [1.0] * 20)
    psi = compute_psi(ref, cur, n_bins=5)
    assert psi >= 0.0


def test_compute_psi_small_arrays_do_not_raise():
    """Very small arrays must not raise even with few unique values."""
    ref = np.array([1.0, 2.0, 3.0])
    cur = np.array([1.0, 2.0, 3.0])
    psi = compute_psi(ref, cur, n_bins=2)
    assert psi >= 0.0


def test_compute_psi_nonoverlapping_distributions_is_finite():
    """
    When the current distribution lies entirely outside the reference bins,
    the epsilon-based formula must return a finite (large) PSI, not inf or nan.
    """
    ref = np.array([1.0] * 50)
    cur = np.array([100.0] * 50)
    psi = compute_psi(ref, cur, n_bins=5)
    assert np.isfinite(psi)
    assert psi >= 0.0
