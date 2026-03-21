from typing import Dict, List

import pandas as pd

# Simple feature registry – can be extended with more sets later.
# NOTE: This module is NOT thread-safe.  Lazy initialization of
# ``entry["features"]`` mutates the global dict without locking.
# This is acceptable because the registry is only used in single-process
# training scripts; the API loads the feature list from model metadata.
FEATURE_SETS: Dict[str, Dict[str, object]] = {
    "v1": {
        "description": "Numeric features from Phase 3 baseline (exclude IDs and target).",
        # The actual list will be inferred at runtime from a reference dataframe if not provided.
        "features": None,
    }
}


def infer_numeric_feature_list(df: pd.DataFrame) -> List[str]:
    """
    Infer numeric feature list from a reference dataframe, excluding IDs and target.

    This mirrors the logic from the Phase 3 notebooks.
    """
    exclude_cols = {"isFraud", "TransactionID", "TransactionDT"}
    numeric_cols = [
        c for c in df.columns if df[c].dtype != "O" and c not in exclude_cols
    ]
    return numeric_cols


def get_feature_list(df: pd.DataFrame, feature_set: str = "v1") -> List[str]:
    """
    Retrieve the ordered list of features for a given feature set.

    If the registry entry has no explicit list yet, infer from df and cache it.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")

    entry = FEATURE_SETS[feature_set]
    if entry["features"] is None:
        entry["features"] = infer_numeric_feature_list(df)
    return list(entry["features"])  # return a copy

