"""Tests for the DeLong test implementation (src/models/delong.py)."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.models.delong import delong_roc_test, delong_roc_variance


@pytest.fixture()
def random_scores():
    rng = np.random.default_rng(42)
    n = 2000
    y = (rng.random(n) < 0.2).astype(int)
    good = rng.normal(loc=y * 1.5, scale=1.0)
    weak = rng.normal(loc=y * 0.5, scale=1.0)
    return y, good, weak


def test_auc_matches_sklearn(random_scores):
    y, good, _ = random_scores
    auc, _ = delong_roc_variance(y, good)
    assert auc == pytest.approx(roc_auc_score(y, good), abs=1e-10)


def test_auc_matches_sklearn_with_ties(random_scores):
    y, good, _ = random_scores
    tied = np.round(good, 1)  # introduce heavy ties
    auc, _ = delong_roc_variance(y, tied)
    assert auc == pytest.approx(roc_auc_score(y, tied), abs=1e-10)


def test_perfect_separation_gives_auc_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auc, var = delong_roc_variance(y, scores)
    assert auc == pytest.approx(1.0)
    assert var >= 0.0


def test_variance_positive_for_imperfect_model(random_scores):
    y, good, _ = random_scores
    _, var = delong_roc_variance(y, good)
    assert var > 0.0


def test_identical_scores_give_zero_delta_p_one(random_scores):
    y, good, _ = random_scores
    result = delong_roc_test(y, good, good)
    assert result.delta == pytest.approx(0.0)
    assert result.z == 0.0
    assert result.p_value == 1.0


def test_monotone_transform_preserves_auc(random_scores):
    """AUC is rank-based: a strictly monotone transform must not change it."""
    y, good, _ = random_scores
    result = delong_roc_test(y, 1.0 / (1.0 + np.exp(-good)), good)
    assert result.delta == pytest.approx(0.0, abs=1e-12)


def test_better_model_detected_as_significant(random_scores):
    y, good, weak = random_scores
    result = delong_roc_test(y, good, weak)
    assert result.auc_a > result.auc_b
    assert result.delta > 0.0
    assert result.p_value < 0.05
    assert result.ci_lower > 0.0


def test_delta_direction_antisymmetric(random_scores):
    y, good, weak = random_scores
    ab = delong_roc_test(y, good, weak)
    ba = delong_roc_test(y, weak, good)
    assert ab.delta == pytest.approx(-ba.delta)
    assert ab.p_value == pytest.approx(ba.p_value)


def test_ci_contains_delta(random_scores):
    y, good, weak = random_scores
    result = delong_roc_test(y, good, weak)
    assert result.ci_lower <= result.delta <= result.ci_upper


def test_single_class_raises():
    y = np.zeros(10, dtype=int)
    scores = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="Both classes"):
        delong_roc_variance(y, scores)


def test_length_mismatch_raises():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="length"):
        delong_roc_test(y, np.zeros(4), np.zeros(3))
