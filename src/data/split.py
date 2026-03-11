from typing import Tuple

import numpy as np
import pandas as pd


def temporal_train_val_split(
    df: pd.DataFrame,
    split_quantile: float,
    target_col: str,
    time_col: str,
    amount_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Temporal split: train on earliest fraction, validate on most recent.
    """
    cutoff = df[time_col].quantile(split_quantile)
    train_mask = df[time_col] < cutoff
    val_mask = ~train_mask

    y_train = df.loc[train_mask, target_col].astype(int)
    y_val = df.loc[val_mask, target_col].astype(int)

    X_train_raw = df.loc[train_mask]
    X_val_raw = df.loc[val_mask]
    val_amount = X_val_raw[amount_col].fillna(0.0)

    return X_train_raw, X_val_raw, y_train, y_val, val_amount

