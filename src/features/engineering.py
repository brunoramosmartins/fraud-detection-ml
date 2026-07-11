"""Feature engineering blocks for the Kaggle extension (Phases 8-9).

All functions are pure: encoders are fit from explicitly passed training
values and applied to any values, so the caller controls the fit/transform
boundary (per-fold training rows in cross-validation; the train partition
for holdout and test scoring). Per project rules, every docstring states
the leakage argument: why the feature is computable at scoring time
without future information.
"""

from typing import Dict, List, Sequence

import numpy as np
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


def add_time_features(transaction_dt: pd.Series) -> pd.DataFrame:
    """Periodic time features from the transaction timestamp (EXP-003).

    Args:
        transaction_dt: ``TransactionDT`` column (seconds from the dataset
            reference instant).

    Returns:
        DataFrame indexed like the input with int32 columns ``tx_hour``
        (0-23) and ``tx_dow`` (0-6).

    Leakage argument:
        Row-local arithmetic on the transaction's own timestamp - no other
        rows, no future information. Deliberately periodic only: an
        absolute time index is NOT exposed, because a raw trend feature
        would let the model split on "late transactions" and extrapolate
        arbitrarily outside the training window.
    """
    return pd.DataFrame(
        {
            "tx_hour": ((transaction_dt // 3600) % 24).astype("int32"),
            "tx_dow": ((transaction_dt // 86400) % 7).astype("int32"),
        },
        index=transaction_dt.index,
    )


def add_amount_features(amount: pd.Series) -> pd.DataFrame:
    """Amount transforms: log scale and decimal part (EXP-003).

    The decimal part (cents) is a known signal in IEEE-CIS: foreign-
    currency transactions converted to USD produce non-round cent values.

    Args:
        amount: ``TransactionAmt`` column.

    Returns:
        DataFrame indexed like the input with float32 columns
        ``amt_log1p`` and ``amt_cents`` (in [0, 1), rounded to 2 decimals).

    Leakage argument:
        Row-local transforms of the transaction's own amount - trivially
        computable at scoring time.
    """
    cents = (amount - np.floor(amount)).round(2)
    return pd.DataFrame(
        {
            "amt_log1p": np.log1p(amount).astype("float32"),
            "amt_cents": cents.astype("float32"),
        },
        index=amount.index,
    )


def normalize_d_columns(
    df: pd.DataFrame,
    transaction_dt: pd.Series,
    d_cols: Sequence[str],
) -> pd.DataFrame:
    """Convert D timedelta columns to time-invariant reference dates (EXP-003).

    The D columns are "days since <some past event>" counters, so their raw
    values grow with absolute time and drift out of the training range on
    the test period. Subtracting the transaction day converts each one to
    "day when the event happened" (a fixed date), which is stable across
    train and test:

    ``{col}_norm = D_col - TransactionDT / 86400``

    Args:
        df: Frame containing the D columns.
        transaction_dt: ``TransactionDT`` column aligned with ``df``.
        d_cols: D columns to normalize (D9 should be excluded - it is an
            hour-of-day fraction, not a day counter).

    Returns:
        DataFrame indexed like ``df`` with float32 ``{col}_norm`` columns.
        NaN inputs stay NaN.

    Leakage argument:
        Row-local arithmetic between two values of the same transaction.
        The subtraction *removes* absolute-time information rather than
        adding it, which is the anti-drift purpose of the transform.
    """
    days = transaction_dt / 86400.0
    out = pd.DataFrame(index=df.index)
    for col in d_cols:
        out[f"{col}_norm"] = (df[col] - days).astype("float32")
    return out


def make_uid(
    df: pd.DataFrame,
    card_col: str = "card1",
    addr_col: str = "addr1",
    dt_col: str = "TransactionDT",
    d1_col: str = "D1",
) -> pd.Series:
    """Build a pseudo-client key (the IEEE-CIS "magic feature", EXP-004).

    Identifies the card/account behind a transaction by combining the card
    id, the billing region, and an inferred account-start day. ``D1`` is
    "days since the card began"; subtracting it from the transaction day
    yields a value that is (approximately) constant for all transactions of
    the same card:

    ``uid = card1 "_" addr1 "_" round(TransactionDT/86400 - D1)``

    Args:
        df: Frame containing the source columns.
        card_col, addr_col, dt_col, d1_col: Column names.

    Returns:
        Plain-string series (object dtype) aligned with ``df``. Each missing
        component is rendered as the literal token ``"NA"`` so the key is
        never null: un-identifiable transactions group by whichever
        components they do have, rather than propagating a missing value or
        fabricating spurious matches.

    Leakage argument:
        Row-local: every component (card1, addr1, TransactionDT, D1) belongs
        to the *same* transaction, so the key uses no other rows and no
        future information. ``TransactionDT/86400 - D1`` cancels absolute
        time, giving a stable per-account reference day computable at
        scoring time from the transaction's own fields. See
        ``exercises/ex03_leakage_and_validation.md`` for the formal argument.
    """
    day = (df[dt_col] / 86400.0).round()
    ref = (day - df[d1_col]).round().astype("Int64").astype("string").fillna("NA")
    card = df[card_col].astype("Int64").astype("string").fillna("NA")
    addr = df[addr_col].astype("Int64").astype("string").fillna("NA")
    return (card + "_" + addr + "_" + ref).astype("object")


def add_uid_aggregates(
    df: pd.DataFrame,
    uid: pd.Series,
    amount_col: str = "TransactionAmt",
) -> pd.DataFrame:
    """Per-UID distributional aggregates (EXP-004).

    Args:
        df: Frame the aggregates are computed over. The caller passes the
            **union of train and test** so a UID seen in both periods gets
            one consistent profile (this is transductive but label-free).
        uid: UID key aligned with ``df`` (from :func:`make_uid`).
        amount_col: Transaction amount column.

    Returns:
        DataFrame indexed like ``df`` with float32 columns: ``uid_count``
        (transactions per UID), ``uid_amt_mean``, ``uid_amt_std`` (0 for
        singletons), and ``uid_amt_ratio`` (this amount / the UID mean —
        flags amounts unusual for the account).

    Leakage argument:
        The aggregates are **label-free** distributional summaries (counts
        and amount moments); no ``isFraud`` value enters them, so computing
        them over the train+test union leaks no target information. It is
        transductive, not temporal leakage: the test transactions are known
        at scoring time (batch scoring), and the private leaderboard - a
        disjoint, later time slice - acts as an independent leak detector
        (a label leak would inflate the public LB and collapse the private
        LB). A real-time serving variant, which cannot see the full union,
        is treated separately in ADR-006.
    """
    work = pd.DataFrame({"uid": uid.to_numpy(), "amt": df[amount_col].to_numpy()})
    grp = work.groupby("uid")["amt"]
    count = grp.transform("count")
    mean = grp.transform("mean")
    std = grp.transform("std").fillna(0.0)
    out = pd.DataFrame(
        {
            "uid_count": count.astype("float32").to_numpy(),
            "uid_amt_mean": mean.astype("float32").to_numpy(),
            "uid_amt_std": std.astype("float32").to_numpy(),
            "uid_amt_ratio": (work["amt"] / mean.replace(0.0, np.nan))
            .astype("float32")
            .to_numpy(),
        },
        index=df.index,
    )
    return out


def combine_columns(df: pd.DataFrame, col1: str, col2: str, sep: str = "_") -> pd.Series:
    """Concatenate two columns into a combined categorical key (EXP-006).

    Mirrors the winning solution's ``encode_CB``: builds a coarser grouping key
    (e.g. ``card1_addr1``) that a single column cannot express.

    Args:
        df: Source frame.
        col1, col2: Column names to combine.
        sep: Separator token.

    Returns:
        Object-dtype string series aligned with ``df``; missing components
        render as the literal missing token (never null).

    Leakage argument:
        Row-local string concatenation — each output depends only on the same
        row's two inputs. No aggregation, no other rows, no future information.
    """
    return (_as_str(df[col1]) + sep + _as_str(df[col2])).astype("object")


def aggregate_group(
    df: pd.DataFrame,
    group_key: str,
    value_cols: "list[str]",
    aggs: "tuple[str, ...]" = ("mean", "std"),
) -> pd.DataFrame:
    """Per-row group aggregates of ``value_cols`` within ``group_key`` (EXP-006).

    Mirrors the winning solution's ``encode_AG``. NaN is left as NaN (LightGBM
    handles it natively) rather than filled with a sentinel — consistent with
    this project's native-NaN choice since EXP-001, and a deliberate deviation
    from Deotte's ``fillna(-1)`` (which suited his XGBoost setup).

    Args:
        df: Frame containing ``group_key`` and ``value_cols``. The caller
            passes the **train+test union** so each group has one consistent
            profile (label-free, transductive).
        group_key: Column name to group by (e.g. ``"uid"``, ``"card1_addr1"``).
        value_cols: Columns to aggregate.
        aggs: Aggregation functions (``"mean"``, ``"std"``, ``"count"``, ...).

    Returns:
        Float32 DataFrame indexed like ``df`` with one column per
        (value_col, agg): ``{value_col}_{group_key}_{agg}``.

    Leakage argument:
        Aggregations are **label-free** (no ``isFraud``), so computing them over
        the train+test union leaks no target. Transductive, not temporal
        leakage: the union is available at batch-scoring time, and the private
        LB (a disjoint later slice) is the independent leak detector. Real-time
        serving needs a client feature store — see ADR-006.
    """
    g = df.groupby(group_key)
    out = pd.DataFrame(index=df.index)
    for col in value_cols:
        transformed = g[col].transform
        for agg in aggs:
            out[f"{col}_{group_key}_{agg}"] = transformed(agg).astype("float32")
    return out


def aggregate_nunique(
    df: pd.DataFrame, group_key: str, value_cols: "list[str]"
) -> pd.DataFrame:
    """Per-row count of distinct ``value_cols`` values within ``group_key`` (EXP-006).

    Mirrors the winning solution's ``encode_AG2``. Captures how many distinct
    emails / devices / values a client (group) touches — a strong fraud signal
    that needs no domain meaning.

    Args:
        df: Frame with ``group_key`` and ``value_cols`` (train+test union).
        group_key: Column name to group by.
        value_cols: Columns whose distinct-value count per group is taken.

    Returns:
        Float32 DataFrame indexed like ``df`` with columns
        ``{group_key}_{value_col}_ct``.

    Leakage argument:
        Same as :func:`aggregate_group`: label-free distributional summary over
        the union; no target information enters.
    """
    g = df.groupby(group_key)
    out = pd.DataFrame(index=df.index)
    for col in value_cols:
        out[f"{group_key}_{col}_ct"] = g[col].transform("nunique").astype("float32")
    return out
