"""Feature selection for temporal stability (EXP-007).

Under temporal drift, a feature that predicts the target well *in the training
period* can fail to generalize to a later test period — inflating internal
validation while hurting the real (private-leaderboard / production) score. This
module provides the statistical selectors the winning IEEE-CIS solutions used to
keep only time-stable features:

- :func:`signed_univariate_auc` — one feature's direction-aware AUC.
- :func:`time_consistency_filter` — drop features whose target association
  flips sign or decays to noise between the first and last training period
  (Deotte's time-consistency test, made explicit).

Adversarial validation (train-vs-test screening) fits a model and lives in the
experiment notebook; the pure, testable statistics live here. See
``docs/kaggle/validation-and-selection-playbook.md``.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def signed_univariate_auc(y: np.ndarray, x: np.ndarray) -> float:
    """Direction-aware univariate ROC-AUC of a single feature.

    Treats the raw feature value as a score. Returns the AUC **without**
    folding to ``max(auc, 1-auc)``, so values below 0.5 signal an inverse
    association — this is what lets :func:`time_consistency_filter` detect a
    sign flip across time.

    NaNs in ``x`` are masked out. If, after masking, fewer than two samples
    remain or only one class is present, the feature is treated as
    uninformative and 0.5 is returned.

    Parameters
    ----------
    y : np.ndarray
        Binary labels, shape ``(n,)``.
    x : np.ndarray
        Feature values, shape ``(n,)``.

    Returns
    -------
    float
        ROC-AUC in ``[0, 1]`` (0.5 = no association / degenerate).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    mask = ~np.isnan(x)
    if mask.sum() < 2:
        return 0.5
    y_m = y[mask]
    if len(np.unique(y_m)) < 2:
        return 0.5
    return float(roc_auc_score(y_m, x[mask]))


def time_consistency_filter(
    features: Dict[str, np.ndarray],
    y: np.ndarray,
    period: np.ndarray,
    flip_margin: float = 0.02,
    decay_strong: float = 0.05,
    decay_weak: float = 0.01,
) -> Tuple[List[str], Dict[str, Tuple[float, float, str]]]:
    """Split features into time-consistent (keep) and time-inconsistent (drop).

    Compares each feature's :func:`signed_univariate_auc` in the **earliest**
    vs the **latest** ``period`` value. A feature is dropped when its
    association with the target is not preserved across time:

    - **flip** — the sign of ``auc - 0.5`` reverses between periods, with both
      magnitudes exceeding ``flip_margin`` (it predicts fraud one way early and
      the opposite way late — actively harmful under drift).
    - **decay** — strong signal in one period (``|auc-0.5| > decay_strong``) but
      noise in the other (``< decay_weak``) — it overfits a single period.

    All other features (consistently useful, or consistently near-0.5 and
    harmless) are kept. This mirrors the winning solutions' behaviour of
    dropping only a handful of genuinely unstable columns, not aggressively
    pruning weak-but-stable ones.

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping of feature name to its values (shape ``(n,)`` each).
    y : np.ndarray
        Binary labels, shape ``(n,)``.
    period : np.ndarray
        Period label per row (e.g. month index), shape ``(n,)``. The min and
        max distinct values define the first and last period.
    flip_margin : float
        Minimum ``|auc-0.5|`` in both periods to count as a sign flip.
    decay_strong, decay_weak : float
        Thresholds defining "strong in one period, noise in the other".

    Returns
    -------
    keep : List[str]
        Feature names retained, in input order.
    dropped : Dict[str, Tuple[float, float, str]]
        Dropped name -> (auc_first, auc_last, reason) where reason is
        ``"flip"`` or ``"decay"``.
    """
    periods = sorted(set(np.asarray(period).tolist()))
    first = np.asarray(period) == periods[0]
    last = np.asarray(period) == periods[-1]

    keep: List[str] = []
    dropped: Dict[str, Tuple[float, float, str]] = {}
    for name, values in features.items():
        values = np.asarray(values, dtype=float)
        a0 = signed_univariate_auc(y[first], values[first])
        a1 = signed_univariate_auc(y[last], values[last])
        d0, d1 = a0 - 0.5, a1 - 0.5
        flip = (d0 > flip_margin and d1 < -flip_margin) or (
            d0 < -flip_margin and d1 > flip_margin
        )
        decay = (abs(d0) > decay_strong and abs(d1) < decay_weak) or (
            abs(d1) > decay_strong and abs(d0) < decay_weak
        )
        if flip:
            dropped[name] = (round(a0, 4), round(a1, 4), "flip")
        elif decay:
            dropped[name] = (round(a0, 4), round(a1, 4), "decay")
        else:
            keep.append(name)
    return keep, dropped


def seen_unseen_masks(
    entity: Sequence, train_mask: np.ndarray, eval_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Split an evaluation set by whether its entity was seen during training.

    Diagnostic for entity-based models under drift: a model can score highly on
    entities it memorised in training while failing on genuinely new ones. This
    returns two boolean masks over ``eval_mask`` rows.

    Parameters
    ----------
    entity : Sequence
        Per-row entity id (e.g. UID), length ``n``.
    train_mask : np.ndarray
        Boolean mask of training rows, shape ``(n,)``.
    eval_mask : np.ndarray
        Boolean mask of evaluation rows, shape ``(n,)``.

    Returns
    -------
    seen : np.ndarray
        ``eval_mask`` rows whose entity also appears in ``train_mask``.
    unseen : np.ndarray
        ``eval_mask`` rows whose entity does not appear in ``train_mask``.
    """
    entity = np.asarray(entity, dtype=object)
    train_entities = set(entity[train_mask].tolist())
    is_seen = np.array([e in train_entities for e in entity], dtype=bool)
    seen = eval_mask & is_seen
    unseen = eval_mask & ~is_seen
    return seen, unseen
