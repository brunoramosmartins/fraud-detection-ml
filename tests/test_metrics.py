import numpy as np

from src.models.metrics import approve_all_baseline_loss, expected_loss, threshold_sweep


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

