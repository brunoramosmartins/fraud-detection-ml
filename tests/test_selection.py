"""Tests for temporal-stability feature selection (src/features/selection.py)."""

import numpy as np
import pytest

from src.features.selection import (
    seen_unseen_masks,
    signed_univariate_auc,
    time_consistency_filter,
)


class TestSignedUnivariateAUC:
    def test_perfect_positive_association(self):
        y = np.array([0, 0, 1, 1])
        x = np.array([0.1, 0.2, 0.8, 0.9])
        assert signed_univariate_auc(y, x) == pytest.approx(1.0)

    def test_perfect_inverse_association_below_half(self):
        y = np.array([0, 0, 1, 1])
        x = np.array([0.9, 0.8, 0.2, 0.1])
        assert signed_univariate_auc(y, x) == pytest.approx(0.0)

    def test_nan_masked_out(self):
        y = np.array([0, 0, 1, 1])
        x = np.array([0.1, np.nan, 0.8, 0.9])
        # remaining rows still perfectly separable
        assert signed_univariate_auc(y, x) == pytest.approx(1.0)

    def test_single_class_returns_half(self):
        y = np.array([1, 1, 1])
        x = np.array([0.1, 0.2, 0.3])
        assert signed_univariate_auc(y, x) == 0.5

    def test_all_nan_returns_half(self):
        y = np.array([0, 1])
        x = np.array([np.nan, np.nan])
        assert signed_univariate_auc(y, x) == 0.5


class TestTimeConsistencyFilter:
    def _period(self, n):
        # first half period 0, second half period 1
        p = np.zeros(n, dtype=int)
        p[n // 2:] = 1
        return p

    def test_consistent_feature_kept(self):
        rng = np.random.default_rng(0)
        n = 400
        y = (rng.random(n) < 0.5).astype(int)
        # consistently positive association in both halves
        x = y + rng.normal(0, 0.3, n)
        period = self._period(n)
        keep, dropped = time_consistency_filter({"good": x}, y, period)
        assert "good" in keep
        assert "good" not in dropped

    def test_flipping_feature_dropped(self):
        n = 400
        y = np.tile([0, 1], n // 2)
        period = self._period(n)
        first, last = period == 0, period == 1
        x = np.empty(n)
        # positive association early, inverse late
        x[first] = np.where(y[first] == 1, 1.0, 0.0)
        x[last] = np.where(y[last] == 1, 0.0, 1.0)
        keep, dropped = time_consistency_filter({"flipper": x}, y, period)
        assert "flipper" in dropped
        assert dropped["flipper"][2] == "flip"

    def test_decaying_feature_dropped(self):
        n = 400
        y = np.tile([0, 1], n // 2)
        period = self._period(n)
        first, last = period == 0, period == 1
        rng = np.random.default_rng(1)
        x = np.empty(n)
        x[first] = np.where(y[first] == 1, 1.0, 0.0)     # strong early
        x[last] = rng.normal(0, 1, last.sum())            # noise late
        keep, dropped = time_consistency_filter({"decayer": x}, y, period)
        assert "decayer" in dropped
        assert dropped["decayer"][2] == "decay"

    def test_harmless_noise_kept(self):
        rng = np.random.default_rng(2)
        n = 400
        y = (rng.random(n) < 0.5).astype(int)
        x = rng.normal(0, 1, n)  # noise in both periods -> harmless, kept
        keep, dropped = time_consistency_filter({"noise": x}, y, self._period(n))
        assert "noise" in keep

    def test_preserves_input_order(self):
        n = 200
        y = (np.arange(n) % 2)
        period = self._period(n)
        feats = {"a": y * 1.0, "b": np.zeros(n), "c": y * 1.0}
        keep, _ = time_consistency_filter(feats, y, period)
        assert keep == [c for c in ["a", "b", "c"] if c in keep]


class TestSeenUnseenMasks:
    def test_splits_by_training_entities(self):
        entity = np.array(["u1", "u2", "u3", "u1", "u4"], dtype=object)
        train_mask = np.array([True, True, False, False, False])
        eval_mask = np.array([False, False, True, True, True])
        seen, unseen = seen_unseen_masks(entity, train_mask, eval_mask)
        # eval rows: idx2(u3 unseen), idx3(u1 seen), idx4(u4 unseen)
        assert seen.tolist() == [False, False, False, True, False]
        assert unseen.tolist() == [False, False, True, False, True]

    def test_masks_are_disjoint_and_cover_eval(self):
        entity = np.array(["a", "b", "c", "a"], dtype=object)
        train_mask = np.array([True, False, False, False])
        eval_mask = np.array([False, True, True, True])
        seen, unseen = seen_unseen_masks(entity, train_mask, eval_mask)
        assert not (seen & unseen).any()
        assert ((seen | unseen) == eval_mask).all()
