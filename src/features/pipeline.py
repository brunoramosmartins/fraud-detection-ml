from typing import List, Tuple

import pandas as pd

from src.features.feature_registry import get_feature_list


def build_features(df: pd.DataFrame, feature_set: str = "v1") -> Tuple[pd.DataFrame, List[str]]:
    """
    Build feature matrix X for the given feature set.

    Responsibilities:
    - select columns from the feature registry
    - apply simple imputations (fillna(0))
    - return X and the ordered feature list actually used
    """
    feature_list = get_feature_list(df, feature_set=feature_set)
    X = df[feature_list].copy()
    X = X.fillna(0.0)
    return X, feature_list

