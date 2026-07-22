"""DeLong test for comparing correlated ROC-AUCs.

Implements the nonparametric approach of DeLong et al. (1988) using the
midrank algorithm of Sun & Xu (2014), which computes the structural
components in O(n log n).

Used by the Phase 8 validation protocol (docs/kaggle/validation-protocol.md):
two models score the same temporal holdout, and the test decides whether the
AUC difference is statistically significant.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DeLongResult:
    """Result of a paired DeLong comparison of two AUCs.

    Attributes
    ----------
    auc_a : float
        AUC of the first score vector (the candidate model).
    auc_b : float
        AUC of the second score vector (the reference model).
    delta : float
        ``auc_a - auc_b``.
    ci_lower, ci_upper : float
        95% confidence interval for ``delta``.
    z : float
        Test statistic ``delta / se(delta)``.
    p_value : float
        Two-sided p-value.
    """

    auc_a: float
    auc_b: float
    delta: float
    ci_lower: float
    ci_upper: float
    z: float
    p_value: float


def _midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks of a 1-D array.

    Parameters
    ----------
    x : np.ndarray
        Input values, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Midranks (ties receive the average rank), shape ``(n,)``, 1-based.
    """
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    n = len(x)
    ranks_sorted = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _structural_components(y_true: np.ndarray, scores: np.ndarray):
    """Compute AUC and DeLong structural components for one score vector.

    The AUC is the Mann-Whitney statistic
    ``P(s_pos > s_neg) + 0.5 * P(s_pos == s_neg)``. The structural
    components (placement values) ``V10`` and ``V01`` are per-observation
    contributions whose empirical covariances yield the AUC variance.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels, shape ``(n,)``.
    scores : np.ndarray
        Predicted scores, shape ``(n,)``.

    Returns
    -------
    auc : float
        Area under the ROC curve.
    v10 : np.ndarray
        Placement values of the positives, shape ``(m,)``.
    v01 : np.ndarray
        Placement values of the negatives, shape ``(n - m,)``.
    """
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        raise ValueError("Both classes must be present to compute an AUC.")

    all_scores = np.concatenate([pos, neg])
    ranks_all = _midrank(all_scores)
    ranks_pos = _midrank(pos)
    ranks_neg = _midrank(neg)

    auc = (ranks_all[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    v10 = (ranks_all[:m] - ranks_pos) / n
    v01 = 1.0 - (ranks_all[m:] - ranks_neg) / m
    return auc, v10, v01


def delong_roc_variance(y_true: np.ndarray, scores: np.ndarray):
    """AUC and its DeLong variance for a single model.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels, shape ``(n,)``.
    scores : np.ndarray
        Predicted scores, shape ``(n,)``.

    Returns
    -------
    auc : float
    variance : float
        ``var(V10)/m + var(V01)/n`` with ddof=1.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    auc, v10, v01 = _structural_components(y_true, scores)
    var = np.var(v10, ddof=1) / len(v10) + np.var(v01, ddof=1) / len(v01)
    return auc, var


def delong_roc_test(
    y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> DeLongResult:
    """Paired DeLong test for two correlated AUCs on the same sample.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels, shape ``(n,)`` — shared by both models.
    scores_a : np.ndarray
        Scores of the candidate model, shape ``(n,)``.
    scores_b : np.ndarray
        Scores of the reference model, shape ``(n,)``.

    Returns
    -------
    DeLongResult
        AUCs, delta with 95% CI, z statistic and two-sided p-value.

    Notes
    -----
    The variance of the difference uses the paired covariance of the
    structural components:

    ``var(delta) = var_a + var_b - 2 * cov(a, b)``

    where each term combines the positive- and negative-class component
    (co)variances scaled by their sample sizes. When the two score vectors
    are identical, ``var(delta)`` collapses to 0; the test then returns
    ``z = 0`` and ``p = 1`` by convention.
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if not (len(y_true) == len(scores_a) == len(scores_b)):
        raise ValueError("y_true, scores_a and scores_b must share one length.")

    auc_a, v10_a, v01_a = _structural_components(y_true, scores_a)
    auc_b, v10_b, v01_b = _structural_components(y_true, scores_b)
    m, n = len(v10_a), len(v01_a)

    var_a = np.var(v10_a, ddof=1) / m + np.var(v01_a, ddof=1) / n
    var_b = np.var(v10_b, ddof=1) / m + np.var(v01_b, ddof=1) / n
    cov_ab = (
        np.cov(v10_a, v10_b, ddof=1)[0, 1] / m
        + np.cov(v01_a, v01_b, ddof=1)[0, 1] / n
    )

    delta = auc_a - auc_b
    var_delta = max(var_a + var_b - 2.0 * cov_ab, 0.0)
    se = np.sqrt(var_delta)

    if se == 0.0:
        z, p = 0.0, 1.0
        ci_lower = ci_upper = delta
    else:
        z = delta / se
        p = 2.0 * stats.norm.sf(abs(z))
        half = stats.norm.ppf(0.975) * se
        ci_lower, ci_upper = delta - half, delta + half

    return DeLongResult(
        auc_a=float(auc_a),
        auc_b=float(auc_b),
        delta=float(delta),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        z=float(z),
        p_value=float(p),
    )
