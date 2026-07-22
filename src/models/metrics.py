from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def expected_loss(
    y_true: Sequence[int],
    proba: Sequence[float],
    amount: Sequence[float],
    c_fp: float,
    threshold: float,
) -> float:
    """Compute Expected Monetary Loss (EML) at a given decision threshold.

    The cost structure is asymmetric: missing a fraudulent transaction
    (false negative) costs the transaction amount, while incorrectly
    flagging a legitimate transaction (false positive) costs a fixed
    operational amount ``c_fp``.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth binary labels (1 = fraud, 0 = legitimate).
    proba : Sequence[float]
        Model-predicted fraud probabilities.
    amount : Sequence[float]
        Transaction amounts, used as the false-negative cost.
    c_fp : float
        Fixed cost incurred per false positive (e.g., manual review cost).
    threshold : float
        Decision threshold; transactions with ``proba >= threshold`` are
        flagged as fraud.

    Returns
    -------
    float
        Total expected monetary loss across all transactions.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    proba_arr = np.asarray(proba)
    amount_arr = np.asarray(amount)

    y_hat = (proba_arr >= threshold).astype(int)
    fn_mask = (y_true_arr == 1) & (y_hat == 0)
    fp_mask = (y_true_arr == 0) & (y_hat == 1)
    loss = (fn_mask * amount_arr + fp_mask * c_fp).sum()
    return float(loss)


def approve_all_baseline_loss(y_true: Sequence[int], amount: Sequence[float]) -> float:
    """Compute the loss under an approve-all policy (no fraud control).

    This represents the worst-case baseline: every fraudulent transaction
    is approved, so the total loss equals the sum of all fraud amounts.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth binary labels (1 = fraud, 0 = legitimate).
    amount : Sequence[float]
        Transaction amounts.

    Returns
    -------
    float
        Total monetary loss when no transactions are flagged.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    amount_arr = np.asarray(amount)
    return float(((y_true_arr == 1) * amount_arr).sum())


def threshold_sweep(
    y_true: Sequence[int],
    proba: Sequence[float],
    amount: Sequence[float],
    c_fp: float,
    thresholds: Optional[Iterable[float]] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Find the decision threshold that minimizes Expected Monetary Loss.

    Evaluates EML at each candidate threshold and returns the one with
    the lowest total cost.  When no thresholds are provided, a default
    grid of 99 evenly spaced values in [0.01, 0.99] is used.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth binary labels (1 = fraud, 0 = legitimate).
    proba : Sequence[float]
        Model-predicted fraud probabilities.
    amount : Sequence[float]
        Transaction amounts (used as false-negative cost).
    c_fp : float
        Fixed cost per false positive.
    thresholds : Iterable[float] or None
        Candidate thresholds to evaluate.  Defaults to
        ``np.linspace(0.01, 0.99, 99)``.

    Returns
    -------
    best_threshold : float
        Threshold that minimizes EML.
    best_loss : float
        EML at the optimal threshold.
    thresholds_arr : np.ndarray
        Array of all evaluated thresholds.
    losses : np.ndarray
        EML at each evaluated threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    thresholds_arr = np.asarray(list(thresholds))
    losses = np.array(
        [expected_loss(y_true, proba, amount, c_fp, t) for t in thresholds_arr]
    )
    best_idx = losses.argmin()
    return float(thresholds_arr[best_idx]), float(losses[best_idx]), thresholds_arr, losses


def precision_fpr_at_threshold(
    y_true: Sequence[int], proba: Sequence[float], threshold: float
) -> Tuple[float, float]:
    """Compute precision and false-positive rate at a given threshold.

    Precision (TP / (TP + FP)) measures the fraction of flagged transactions
    that are actual fraud — i.e., the density of fraud in the alert queue.
    Some fraud-detection teams call this the "Fraud Detection Rate."

    FPR (FP / (FP + TN)) measures how many legitimate transactions are
    incorrectly flagged.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth binary labels (1 = fraud, 0 = legitimate).
    proba : Sequence[float]
        Model-predicted fraud probabilities.
    threshold : float
        Decision threshold; transactions with ``proba >= threshold`` are
        flagged as fraud.

    Returns
    -------
    precision : float
        TP / (TP + FP), or 0.0 if no transactions are flagged.
    fpr : float
        FP / (FP + TN), or 0.0 if there are no legitimate transactions.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(precision), float(fpr)


def compute_classification_metrics(
    y_true: Sequence[int],
    proba: Sequence[float],
    amount: Sequence[float],
    c_fp: float,
    thresholds: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    """Compute all classification and cost-sensitive metrics in one call.

    Returns a dictionary with ROC-AUC, PR-AUC, baseline loss,
    cost-optimal threshold, best loss, expected loss reduction,
    precision, and false-positive rate.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth binary labels (1 = fraud, 0 = legitimate).
    proba : Sequence[float]
        Model-predicted fraud probabilities.
    amount : Sequence[float]
        Transaction amounts (used as false-negative cost).
    c_fp : float
        Fixed cost per false positive.
    thresholds : Iterable[float] or None
        Candidate thresholds for the sweep.

    Returns
    -------
    Dict[str, float]
        Keys: ``roc_auc``, ``pr_auc``, ``baseline_loss``,
        ``best_threshold``, ``best_loss``, ``expected_loss_reduction``,
        ``precision``, ``fpr``.
    """
    roc = roc_auc_score(y_true, proba)
    pr = average_precision_score(y_true, proba)
    baseline = approve_all_baseline_loss(y_true, amount)
    best_t, best_loss, _, _ = threshold_sweep(y_true, proba, amount, c_fp, thresholds)
    precision, fpr = precision_fpr_at_threshold(y_true, proba, best_t)
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "baseline_loss": baseline,
        "best_threshold": best_t,
        "best_loss": best_loss,
        "expected_loss_reduction": baseline - best_loss,
        "precision": precision,
        "fpr": fpr,
    }

