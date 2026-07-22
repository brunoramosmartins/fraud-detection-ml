import numpy as np
import pytest

from src.models.metrics import expected_calibration_error, reliability_table


def _calibrated_sample(n: int = 20000, seed: int = 42):
    """Scores drawn so that P(y=1 | p) == p exactly (perfectly calibrated)."""
    rng = np.random.default_rng(seed)
    proba = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(size=n) < proba).astype(int)
    return y, proba


def test_reliability_table_shapes_and_counts():
    y, proba = _calibrated_sample()
    tbl = reliability_table(y, proba, n_bins=10, strategy="quantile")
    assert set(tbl) == {"bin_lower", "bin_upper", "count", "mean_predicted", "observed_rate"}
    assert tbl["count"].sum() == len(y)
    # quantile bins on continuous scores are near-equal-count
    assert tbl["count"].min() > 0.5 * len(y) / 10


def test_reliability_table_uniform_strategy():
    y, proba = _calibrated_sample()
    tbl = reliability_table(y, proba, n_bins=10, strategy="uniform")
    assert tbl["count"].sum() == len(y)
    assert (tbl["bin_upper"] - tbl["bin_lower"]).max() == pytest.approx(0.1)


def test_reliability_table_rejects_unknown_strategy():
    y, proba = _calibrated_sample(n=100)
    with pytest.raises(ValueError, match="strategy"):
        reliability_table(y, proba, strategy="magic")


def test_calibrated_scores_have_low_ece():
    y, proba = _calibrated_sample()
    assert expected_calibration_error(y, proba, n_bins=10) < 0.02


def test_miscalibrated_scores_have_high_ece():
    y, proba = _calibrated_sample()
    # Systematic overconfidence: squash scores toward 1
    distorted = np.sqrt(proba)
    ece_raw = expected_calibration_error(y, proba, n_bins=10)
    ece_bad = expected_calibration_error(y, distorted, n_bins=10)
    assert ece_bad > ece_raw + 0.05


def test_ece_zero_for_constant_perfect_rate():
    # Every prediction 0.5, observed rate exactly 0.5 -> ECE == 0
    y = np.array([0, 1] * 500)
    proba = np.full(1000, 0.5)
    assert expected_calibration_error(y, proba) == pytest.approx(0.0)


def test_reliability_table_concentrated_scores_do_not_crash():
    """Fraud-like distribution: scores concentrated near 0 with ties."""
    rng = np.random.default_rng(0)
    proba = np.concatenate([np.zeros(5000), rng.beta(1, 50, size=5000)])
    y = (rng.uniform(size=10000) < proba).astype(int)
    tbl = reliability_table(y, proba, n_bins=10, strategy="quantile")
    # Tied quantile edges collapse; table still covers every observation
    assert tbl["count"].sum() == len(y)
