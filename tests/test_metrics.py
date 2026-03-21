import numpy as np

from src.models.metrics import (
    approve_all_baseline_loss,
    compute_classification_metrics,
    expected_loss,
    precision_fpr_at_threshold,
    threshold_sweep,
)


def test_expected_loss_and_baseline_simple_case():
    y_true = [0, 1]
    proba = [0.2, 0.9]
    amount = [10.0, 100.0]
    c_fp = 5.0

    baseline = approve_all_baseline_loss(y_true, amount)
    assert np.isclose(baseline, 100.0)

    loss_t0 = expected_loss(y_true, proba, amount, c_fp, threshold=0.5)
    # First sample not flagged (correct), second flagged (correct) => no cost
    assert np.isclose(loss_t0, 0.0)

    loss_t1 = expected_loss(y_true, proba, amount, c_fp, threshold=0.95)
    # Both unflagged => one FN cost (100)
    assert np.isclose(loss_t1, 100.0)


def test_threshold_sweep_finds_better_than_baseline():
    y_true = [0, 1, 1]
    proba = [0.1, 0.9, 0.8]
    amount = [10.0, 100.0, 200.0]
    c_fp = 5.0

    baseline = approve_all_baseline_loss(y_true, amount)
    best_t, best_loss, _, _ = threshold_sweep(y_true, proba, amount, c_fp)

    assert best_loss <= baseline
    assert 0.0 < best_t < 1.0


def test_precision_fpr_at_threshold():
    # 2 fraud (proba 0.9, 0.3), 2 legit (proba 0.1, 0.6)
    y_true = [1, 0, 1, 0]
    proba = [0.9, 0.1, 0.3, 0.6]

    # threshold=0.5 → flags indices 0 (TP) and 3 (FP)
    precision, fpr = precision_fpr_at_threshold(y_true, proba, threshold=0.5)
    assert np.isclose(precision, 0.5)  # 1 TP / (1 TP + 1 FP)
    assert np.isclose(fpr, 0.5)  # 1 FP / (1 FP + 1 TN)


def test_precision_fpr_when_nothing_flagged():
    y_true = [0, 1]
    proba = [0.1, 0.2]
    precision, fpr = precision_fpr_at_threshold(y_true, proba, threshold=0.99)
    assert precision == 0.0
    assert fpr == 0.0


def test_compute_classification_metrics_returns_all_keys():
    y_true = [0, 1, 1, 0, 0]
    proba = [0.1, 0.9, 0.8, 0.2, 0.3]
    amount = [10.0, 100.0, 200.0, 50.0, 30.0]

    result = compute_classification_metrics(y_true, proba, amount, c_fp=5.0)
    expected_keys = {
        "roc_auc",
        "pr_auc",
        "baseline_loss",
        "best_threshold",
        "best_loss",
        "expected_loss_reduction",
        "precision",
        "fpr",
    }
    assert set(result.keys()) == expected_keys
    assert result["expected_loss_reduction"] >= 0
    assert 0.0 <= result["roc_auc"] <= 1.0

