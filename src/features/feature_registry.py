from typing import Dict, List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Feature set "v2" — explicit block definition (ADR-006).
#
# v2 is the serving-feasible subset of the Phase 8 Kaggle feature blocks:
# the EXP-003 configuration (categorical encodings + time/amount/D-norm on
# top of the numeric base), served with LightGBM and native NaN handling.
# The definitions below are constants, not runtime inference: they pin the
# exact columns each block touches so the training pipeline and the Kaggle
# notebooks provably build the same features.
# ---------------------------------------------------------------------------

# Email-domain columns split into provider/suffix before encoding
# (source column, output prefix).
V2_EMAIL_SPLITS: List[Tuple[str, str]] = [
    ("P_emaildomain", "P_email"),
    ("R_emaildomain", "R_email"),
]

# Numeric-dtype categoricals that additionally get frequency encoding
# (high-cardinality ids where relative frequency is the signal).
V2_FREQ_NUMERIC_CATS: List[str] = ["card1", "card2", "card3", "card5", "addr1", "addr2"]

# D timedelta columns normalized to fixed reference dates
# (D9 excluded: it is an hour-of-day fraction, not a day counter).
V2_D_NORM_COLS: List[str] = [f"D{i}" for i in range(1, 16) if i != 9]

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
    },
    "v2": {
        "description": (
            "Serving-feasible Kaggle blocks (ADR-006): numeric base with native "
            "NaN + categorical label/frequency encodings + email split + "
            "time/amount features + D-column normalization. Built by "
            "src.features.pipeline.FeatureBuilderV2, which owns the fitted "
            "encoder state; the final list lives in the fitted builder and in "
            "the artifact metadata."
        ),
        # v2 output columns depend on fitted encoder state; use FeatureBuilderV2.
        "features": None,
    },
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
    if feature_set == "v2":
        raise ValueError(
            "Feature set 'v2' requires fitted encoder state and cannot be "
            "expressed as a static column list; build it with "
            "src.features.pipeline.FeatureBuilderV2."
        )

    entry = FEATURE_SETS[feature_set]
    if entry["features"] is None:
        entry["features"] = infer_numeric_feature_list(df)
    return list(entry["features"])  # return a copy

