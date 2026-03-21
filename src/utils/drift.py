import numpy as np

_PSI_EPSILON = 1e-4  # Minimum bin proportion to avoid log(0); standard PSI convention.


def compute_psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between a reference and a current
    distribution using quantile-based binning derived from the reference.

    PSI interpretation (common industry thresholds):
        < 0.10  — stable, no meaningful shift
        0.10 – 0.20 — slight shift, worth monitoring
        > 0.20  — significant drift, consider retraining

    Args:
        ref:    1-D array representing the reference (e.g. training) distribution.
        cur:    1-D array representing the current (e.g. production) distribution.
        n_bins: Number of quantile bins derived from ``ref``.

    Returns:
        PSI value in [0, ∞).  Returns 0.0 when ``ref`` has fewer than two
        unique bin edges after deduplication.
    """
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)

    if np.allclose(ref, ref[0]):
        # Constant reference: fall back to fixed-width bins centred on the value.
        bin_edges = np.linspace(ref[0] - 0.5, ref[0] + 0.5, n_bins + 1)
    else:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(ref, quantiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) <= 1:
            return 0.0

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_total = ref_counts.sum()
    cur_total = cur_counts.sum()

    # Guard against a zero denominator: this happens when all current observations
    # fall outside the reference bin range (extreme drift).  Dividing by zero would
    # produce NaN, which np.where(...== 0) cannot catch.
    ref_perc = ref_counts / ref_total if ref_total > 0 else np.zeros(len(ref_counts), dtype=float)
    cur_perc = cur_counts / cur_total if cur_total > 0 else np.zeros(len(cur_counts), dtype=float)

    # Replace zero proportions with epsilon to keep the formula defined for
    # all bins, including those that receive no observations in the current
    # window.  Skipping zero-bins instead would silently underestimate drift.
    ref_perc = np.where(ref_perc == 0, _PSI_EPSILON, ref_perc)
    cur_perc = np.where(cur_perc == 0, _PSI_EPSILON, cur_perc)

    psi = float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))
    return psi
