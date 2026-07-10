"""Compute the DeLong test between two Kaggle-experiment holdout artifacts.

Each experiment notebook saves ``holdout_pred_expNNN.csv`` with columns
``TransactionID, y_true, score``. This script aligns two such files by
``TransactionID`` and reports the DeLong comparison (candidate vs reference)
using the tested :func:`src.models.delong.delong_roc_test`.

This is the off-notebook DeLong step mandated by
``docs/kaggle/validation-protocol.md`` (amended 2026-07-10).

Usage:
    python scripts/kaggle_delong.py NEW.csv OLD.csv
    python scripts/kaggle_delong.py holdout_pred_exp002.csv holdout_pred_exp001.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.models.delong import delong_roc_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_csv", type=Path, help="candidate holdout artifact")
    parser.add_argument("old_csv", type=Path, help="reference (predecessor) holdout artifact")
    args = parser.parse_args()

    new = pd.read_csv(args.new_csv)
    old = pd.read_csv(args.old_csv)
    merged = new.merge(old, on="TransactionID", suffixes=("_new", "_old"))
    if len(merged) != len(new):
        raise ValueError(
            f"row mismatch after align: new={len(new)} merged={len(merged)} "
            "- holdout sets differ (were both from the same Scheme-A split?)"
        )
    if not (merged["y_true_new"] == merged["y_true_old"]).all():
        raise ValueError("labels differ between the two artifacts - alignment bug")

    result = delong_roc_test(
        merged["y_true_new"].to_numpy(),
        merged["score_new"].to_numpy(),
        merged["score_old"].to_numpy(),
    )
    print(f"n holdout rows      : {len(merged):,}")
    print(f"AUC new ({args.new_csv.stem}): {result.auc_a:.4f}")
    print(f"AUC old ({args.old_csv.stem}): {result.auc_b:.4f}")
    print(f"delta AUC           : {result.delta:+.4f}")
    print(f"95% CI              : [{result.ci_lower:+.4f}, {result.ci_upper:+.4f}]")
    print(f"z                   : {result.z:.2f}")
    print(f"p-value (two-sided) : {result.p_value:.3e}")
    print(f"significant @0.05   : {result.p_value < 0.05 and result.delta > 0}")


if __name__ == "__main__":
    main()
