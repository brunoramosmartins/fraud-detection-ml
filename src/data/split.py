from typing import Tuple

import pandas as pd


def temporal_train_val_split(
    df: pd.DataFrame,
    split_quantile: float,
    target_col: str,
    time_col: str,
    amount_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split a dataset temporally into train (past) and validation (future).

    Uses a quantile of ``time_col`` as the cutoff: rows before the cutoff
    form the training set, rows at or after form the validation set.
    This prevents data leakage from future transactions into training.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including target, time, and amount columns.
    split_quantile : float
        Fraction of the time range used for training (e.g., 0.8 = 80 %
        oldest transactions).
    target_col : str
        Name of the binary target column.
    time_col : str
        Name of the temporal ordering column.
    amount_col : str
        Name of the transaction amount column (returned for cost
        computation on the validation set).

    Returns
    -------
    X_train_raw : pd.DataFrame
        Training rows (all columns).
    X_val_raw : pd.DataFrame
        Validation rows (all columns).
    y_train : pd.Series
        Training labels.
    y_val : pd.Series
        Validation labels.
    val_amount : pd.Series
        Transaction amounts for the validation set (NaN filled with 0).
    """
    missing = [c for c in (target_col, time_col, amount_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from DataFrame: {missing}")
    cutoff = df[time_col].quantile(split_quantile)
    train_mask = df[time_col] < cutoff
    val_mask = ~train_mask

    y_train = df.loc[train_mask, target_col].astype(int)
    y_val = df.loc[val_mask, target_col].astype(int)

    X_train_raw = df.loc[train_mask]
    X_val_raw = df.loc[val_mask]
    val_amount = X_val_raw[amount_col].fillna(0.0)

    return X_train_raw, X_val_raw, y_train, y_val, val_amount

