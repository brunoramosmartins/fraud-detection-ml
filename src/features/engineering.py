"""Feature engineering blocks for the Kaggle extension (Phases 8-9).

All functions are pure: encoders are fit from explicitly passed training
values and applied to any values, so the caller controls the fit/transform
boundary (per-fold training rows in cross-validation; the train partition
for holdout and test scoring). Per project rules, every docstring states
the leakage argument: why the feature is computable at scoring time
without future information.
"""

from typing import Dict, List

import pandas as pd

MISSING_TOKEN = "__missing__"


def _as_str(values: pd.Series) -> pd.Series:
    """Normalize a column to strings with an explicit missing token."""
    return (
        values.astype("object")
        .where(values.notna(), MISSING_TOKEN)
        .astype(str)
    )


def frequency_encode(train_values: pd.Series, values: pd.Series) -> pd.Series:
    """Encode categories by their relative frequency in the training data.

    Args:
        train_values: Column the frequency table is fit on (training rows
            only).
        values: Column to encode (may be the same series, a validation
            fold, the holdout, or the test set).

    Returns:
        Float32 series aligned with ``values``: each category mapped to its
        relative frequency among ``train_values``. Categories never seen in
        training map to 0.0. Missing values are treated as a category of
        their own.

    Leakage argument:
        The frequency table is computed exclusively from ``train_values``
        (past data). At scoring time each transaction is mapped through
        that fixed table; no information from the scored rows - present or
        future - enters the encoding. The 0.0 fallback for unseen
        categories is exactly the behavior available in production for a
        never-observed value.
    """
    freq = _as_str(train_values).value_counts(normalize=True)
    return _as_str(values).map(freq).fillna(0.0).astype("float32")


def label_encode(train_values: pd.Series, values: pd.Series) -> pd.Series:
    """Encode categories as integer codes from a training-fit dictionary.

    Args:
        train_values: Column the category dictionary is fit on (training
            rows only). Categories are sorted before code assignment so the
            mapping is deterministic across runs.
        values: Column to encode.

    Returns:
        Int32 series aligned with ``values``. Categories absent from
        training (including a missing value when training had none) map
        to -1.

    Leakage argument:
        The category-to-code dictionary is built from training values only
        and frozen. Applying a fixed dictionary row-by-row requires no
        knowledge of other scored rows or of the future; unseen categories
        degrade gracefully to the -1 sentinel, the same behavior a serving
        system exhibits for a brand-new category.
    """
    cats: Dict[str, int] = {
        v: i for i, v in enumerate(sorted(_as_str(train_values).unique()))
    }
    return _as_str(values).map(cats).fillna(-1).astype("int32")


def split_email_domain(values: pd.Series, prefix: str) -> pd.DataFrame:
    """Split an email domain into provider and suffix columns.

    ``"gmail.com"`` becomes provider ``"gmail"`` and suffix ``"com"``;
    ``"mail.co.uk"`` becomes ``"mail"`` / ``"uk"``. Missing domains yield
    the missing token in both columns.

    Args:
        values: Email domain column (e.g. ``P_emaildomain``).
        prefix: Prefix for the output column names.

    Returns:
        DataFrame indexed like ``values`` with string columns
        ``{prefix}_provider`` and ``{prefix}_suffix`` (encode them with
        :func:`label_encode` / :func:`frequency_encode` afterwards).

    Leakage argument:
        Row-local string operation: each output value depends only on the
        same row's input value. No aggregation, no other rows, no future
        information - leak-free by construction.
    """
    s = _as_str(values)
    parts = s.str.split(".")
    return pd.DataFrame(
        {
            f"{prefix}_provider": parts.str[0],
            f"{prefix}_suffix": parts.str[-1],
        },
        index=values.index,
    )


def build_categorical_block(
    train_df: pd.DataFrame,
    df: pd.DataFrame,
    label_cols: List[str],
    freq_cols: List[str],
) -> pd.DataFrame:
    """Build the H2 categorical feature block (EXP-002).

    Args:
        train_df: Training rows the encoders are fit on.
        df: Rows to encode (training, validation, holdout, or test).
        label_cols: Columns to label-encode (suffix ``_le``).
        freq_cols: Columns to frequency-encode (suffix ``_freq``). A column
            may appear in both lists.

    Returns:
        DataFrame indexed like ``df`` containing only the new encoded
        columns, ready to be concatenated to the numeric feature matrix.

    Leakage argument:
        Pure orchestration of :func:`label_encode` and
        :func:`frequency_encode`; both are fit on ``train_df`` only, so the
        block inherits their scoring-time computability.
    """
    out = pd.DataFrame(index=df.index)
    for col in label_cols:
        out[f"{col}_le"] = label_encode(train_df[col], df[col])
    for col in freq_cols:
        out[f"{col}_freq"] = frequency_encode(train_df[col], df[col])
    return out
