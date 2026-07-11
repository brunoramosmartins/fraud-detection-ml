"""Tests for the categorical encoding blocks (src/features/engineering.py)."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    MISSING_TOKEN,
    add_amount_features,
    add_time_features,
    add_uid_aggregates,
    aggregate_group,
    aggregate_nunique,
    build_categorical_block,
    combine_columns,
    frequency_encode,
    label_encode,
    make_uid,
    normalize_d_columns,
    split_email_domain,
)


@pytest.fixture()
def train_col():
    return pd.Series(["a", "a", "a", "b", np.nan, "b", "c", "a"])


class TestFrequencyEncode:
    def test_frequencies_computed_on_train_only(self, train_col):
        scoring = pd.Series(["a", "b", "c"])
        encoded = frequency_encode(train_col, scoring)
        assert encoded.iloc[0] == pytest.approx(4 / 8)  # 'a' appears 4/8 in train
        assert encoded.iloc[1] == pytest.approx(2 / 8)
        assert encoded.iloc[2] == pytest.approx(1 / 8)

    def test_unseen_category_maps_to_zero(self, train_col):
        encoded = frequency_encode(train_col, pd.Series(["zzz"]))
        assert encoded.iloc[0] == 0.0

    def test_missing_is_its_own_category(self, train_col):
        encoded = frequency_encode(train_col, pd.Series([np.nan]))
        assert encoded.iloc[0] == pytest.approx(1 / 8)  # one NaN in train

    def test_scoring_distribution_does_not_affect_encoding(self, train_col):
        skewed = pd.Series(["c"] * 100)
        encoded = frequency_encode(train_col, skewed)
        assert encoded.to_numpy() == pytest.approx(1 / 8)

    def test_dtype_is_float32(self, train_col):
        assert frequency_encode(train_col, train_col).dtype == np.float32


class TestLabelEncode:
    def test_mapping_is_deterministic_and_sorted(self, train_col):
        encoded = label_encode(train_col, pd.Series(["a", "b", "c"]))
        # sorted unique train categories: __missing__, a, b, c
        assert encoded.tolist() == [1, 2, 3]

    def test_unseen_category_maps_to_minus_one(self, train_col):
        encoded = label_encode(train_col, pd.Series(["zzz"]))
        assert encoded.iloc[0] == -1

    def test_missing_seen_in_train_gets_a_code(self, train_col):
        encoded = label_encode(train_col, pd.Series([np.nan]))
        assert encoded.iloc[0] == 0  # MISSING_TOKEN sorts first

    def test_missing_unseen_in_train_maps_to_minus_one(self):
        train = pd.Series(["a", "b"])
        encoded = label_encode(train, pd.Series([np.nan]))
        assert encoded.iloc[0] == -1

    def test_dtype_is_int32(self, train_col):
        assert label_encode(train_col, train_col).dtype == np.int32

    def test_numeric_categories_are_stable(self):
        train = pd.Series([1111.0, 2222.0, np.nan])
        encoded = label_encode(train, pd.Series([2222.0, 3333.0]))
        assert encoded.iloc[0] >= 0
        assert encoded.iloc[1] == -1


class TestSplitEmailDomain:
    def test_two_part_domain(self):
        out = split_email_domain(pd.Series(["gmail.com"]), "P")
        assert out.loc[0, "P_provider"] == "gmail"
        assert out.loc[0, "P_suffix"] == "com"

    def test_multi_part_domain(self):
        out = split_email_domain(pd.Series(["mail.co.uk"]), "P")
        assert out.loc[0, "P_provider"] == "mail"
        assert out.loc[0, "P_suffix"] == "uk"

    def test_missing_yields_missing_token(self):
        out = split_email_domain(pd.Series([np.nan]), "P")
        assert out.loc[0, "P_provider"] == MISSING_TOKEN
        assert out.loc[0, "P_suffix"] == MISSING_TOKEN

    def test_index_preserved(self):
        s = pd.Series(["gmail.com", "yahoo.com"], index=[10, 20])
        out = split_email_domain(s, "R")
        assert list(out.index) == [10, 20]


class TestBuildCategoricalBlock:
    def test_column_names_and_shape(self):
        train = pd.DataFrame({"cat": ["a", "b"], "card": [1.0, 2.0]})
        block = build_categorical_block(
            train, train, label_cols=["cat"], freq_cols=["cat", "card"]
        )
        assert sorted(block.columns) == ["card_freq", "cat_freq", "cat_le"]
        assert len(block) == 2

    def test_fit_on_train_apply_on_test(self):
        train = pd.DataFrame({"cat": ["a", "a", "b"]})
        test = pd.DataFrame({"cat": ["b", "zzz"]}, index=[100, 101])
        block = build_categorical_block(train, test, ["cat"], ["cat"])
        assert block.loc[101, "cat_le"] == -1
        assert block.loc[101, "cat_freq"] == 0.0
        assert list(block.index) == [100, 101]

    def test_only_new_columns_returned(self):
        train = pd.DataFrame({"cat": ["a"], "other": [1]})
        block = build_categorical_block(train, train, ["cat"], [])
        assert list(block.columns) == ["cat_le"]


class TestAddTimeFeatures:
    def test_hour_and_dow_computed(self):
        # 90000 s = 1 day + 1 h -> hour 1, dow 1
        out = add_time_features(pd.Series([0, 90000]))
        assert out["tx_hour"].tolist() == [0, 1]
        assert out["tx_dow"].tolist() == [0, 1]

    def test_hour_wraps_at_24(self):
        out = add_time_features(pd.Series([24 * 3600]))
        assert out.loc[0, "tx_hour"] == 0

    def test_dow_wraps_at_7(self):
        out = add_time_features(pd.Series([7 * 86400]))
        assert out.loc[0, "tx_dow"] == 0

    def test_no_absolute_time_column(self):
        out = add_time_features(pd.Series([123456]))
        assert set(out.columns) == {"tx_hour", "tx_dow"}


class TestAddAmountFeatures:
    def test_log_and_cents(self):
        out = add_amount_features(pd.Series([99.95]))
        assert out.loc[0, "amt_log1p"] == pytest.approx(np.log1p(99.95), abs=1e-6)
        assert out.loc[0, "amt_cents"] == pytest.approx(0.95, abs=1e-6)

    def test_round_amount_has_zero_cents(self):
        out = add_amount_features(pd.Series([50.0]))
        assert out.loc[0, "amt_cents"] == 0.0


class TestNormalizeDColumns:
    def test_reference_date_is_time_invariant(self):
        # Same event date seen from two transaction days must normalize equal:
        # event on day 10 -> D1 = 5 at day 15, D1 = 20 at day 30.
        df = pd.DataFrame({"D1": [5.0, 20.0]})
        dt = pd.Series([15 * 86400, 30 * 86400])
        out = normalize_d_columns(df, dt, ["D1"])
        assert out["D1_norm"].iloc[0] == pytest.approx(out["D1_norm"].iloc[1])
        assert out["D1_norm"].iloc[0] == pytest.approx(-10.0)

    def test_nan_stays_nan(self):
        df = pd.DataFrame({"D1": [np.nan]})
        out = normalize_d_columns(df, pd.Series([86400]), ["D1"])
        assert np.isnan(out.loc[0, "D1_norm"])

    def test_column_naming(self):
        df = pd.DataFrame({"D1": [1.0], "D4": [2.0]})
        out = normalize_d_columns(df, pd.Series([0]), ["D1", "D4"])
        assert list(out.columns) == ["D1_norm", "D4_norm"]


class TestMakeUid:
    def test_same_account_gets_same_uid(self):
        # Same card1/addr1; two transactions 5 days apart with D1 rising by 5
        # -> identical account-start day -> identical UID.
        df = pd.DataFrame(
            {
                "card1": [1000, 1000],
                "addr1": [200, 200],
                "TransactionDT": [10 * 86400, 15 * 86400],
                "D1": [3.0, 8.0],
            }
        )
        uid = make_uid(df)
        assert uid.iloc[0] == uid.iloc[1]

    def test_different_card_differs(self):
        df = pd.DataFrame(
            {
                "card1": [1000, 2000],
                "addr1": [200, 200],
                "TransactionDT": [10 * 86400, 10 * 86400],
                "D1": [3.0, 3.0],
            }
        )
        uid = make_uid(df)
        assert uid.iloc[0] != uid.iloc[1]

    def test_missing_component_does_not_crash(self):
        df = pd.DataFrame(
            {
                "card1": [1000, np.nan],
                "addr1": [200, 200],
                "TransactionDT": [10 * 86400, 10 * 86400],
                "D1": [3.0, np.nan],
            }
        )
        uid = make_uid(df)
        assert len(uid) == 2
        assert uid.iloc[0] != uid.iloc[1]


class TestAddUidAggregates:
    def test_count_and_mean(self):
        df = pd.DataFrame({"TransactionAmt": [100.0, 300.0, 50.0]})
        uid = pd.Series(["a", "a", "b"])
        out = add_uid_aggregates(df, uid)
        assert out["uid_count"].tolist() == [2.0, 2.0, 1.0]
        assert out["uid_amt_mean"].iloc[0] == pytest.approx(200.0)
        assert out["uid_amt_mean"].iloc[2] == pytest.approx(50.0)

    def test_singleton_std_is_zero(self):
        df = pd.DataFrame({"TransactionAmt": [50.0]})
        out = add_uid_aggregates(df, pd.Series(["b"]))
        assert out["uid_amt_std"].iloc[0] == 0.0

    def test_amount_ratio(self):
        df = pd.DataFrame({"TransactionAmt": [100.0, 300.0]})
        out = add_uid_aggregates(df, pd.Series(["a", "a"]))
        # mean = 200 -> ratios 0.5 and 1.5
        assert out["uid_amt_ratio"].iloc[0] == pytest.approx(0.5)
        assert out["uid_amt_ratio"].iloc[1] == pytest.approx(1.5)

    def test_index_preserved(self):
        df = pd.DataFrame({"TransactionAmt": [1.0, 2.0]}, index=[7, 9])
        out = add_uid_aggregates(df, pd.Series(["a", "a"], index=[7, 9]))
        assert list(out.index) == [7, 9]

    def test_no_label_column_used(self):
        # Aggregates must be identical whether or not a label column exists.
        df = pd.DataFrame({"TransactionAmt": [10.0, 20.0], "isFraud": [1, 0]})
        uid = pd.Series(["a", "a"])
        out = add_uid_aggregates(df, uid)
        assert out["uid_amt_mean"].tolist() == [15.0, 15.0]


class TestCombineColumns:
    def test_concatenates(self):
        df = pd.DataFrame({"card1": [1000, 2000], "addr1": [10, 20]})
        out = combine_columns(df, "card1", "addr1")
        assert out.tolist() == ["1000_10", "2000_20"]

    def test_missing_uses_token(self):
        df = pd.DataFrame({"card1": [1000, np.nan], "addr1": [10, 20]})
        out = combine_columns(df, "card1", "addr1")
        assert out.iloc[1] == f"{MISSING_TOKEN}_20"

    def test_distinguishes_pairs(self):
        # (1,2) and (12, "") must not collide -> separator matters
        df = pd.DataFrame({"a": ["1", "12"], "b": ["2", ""]})
        out = combine_columns(df, "a", "b")
        assert out.iloc[0] != out.iloc[1]


class TestAggregateGroup:
    def test_mean_and_std_over_group(self):
        df = pd.DataFrame(
            {"uid": ["a", "a", "b"], "TransactionAmt": [100.0, 300.0, 50.0]}
        )
        out = aggregate_group(df, "uid", ["TransactionAmt"], aggs=("mean", "std"))
        assert out["TransactionAmt_uid_mean"].tolist() == [200.0, 200.0, 50.0]
        # group b is a singleton -> std is NaN (left as NaN by design)
        assert np.isnan(out["TransactionAmt_uid_std"].iloc[2])

    def test_column_naming(self):
        df = pd.DataFrame({"uid": ["a"], "C1": [1.0], "C2": [2.0]})
        out = aggregate_group(df, "uid", ["C1", "C2"], aggs=("mean",))
        assert sorted(out.columns) == ["C1_uid_mean", "C2_uid_mean"]

    def test_dtype_float32(self):
        df = pd.DataFrame({"uid": ["a", "a"], "x": [1.0, 3.0]})
        out = aggregate_group(df, "uid", ["x"], aggs=("mean",))
        assert out["x_uid_mean"].dtype == np.float32

    def test_label_free(self):
        df = pd.DataFrame(
            {"uid": ["a", "a"], "x": [1.0, 3.0], "isFraud": [1, 0]}
        )
        out = aggregate_group(df, "uid", ["x"], aggs=("mean",))
        assert out["x_uid_mean"].tolist() == [2.0, 2.0]


class TestAggregateNunique:
    def test_distinct_count_per_group(self):
        df = pd.DataFrame(
            {"uid": ["a", "a", "a", "b"], "email": ["x", "x", "y", "z"]}
        )
        out = aggregate_nunique(df, "uid", ["email"])
        # group a touches 2 distinct emails, b touches 1
        assert out["uid_email_ct"].tolist() == [2.0, 2.0, 2.0, 1.0]

    def test_column_naming(self):
        df = pd.DataFrame({"uid": ["a"], "dev": ["d1"], "dist": [1.0]})
        out = aggregate_nunique(df, "uid", ["dev", "dist"])
        assert sorted(out.columns) == ["uid_dev_ct", "uid_dist_ct"]

    def test_dtype_float32(self):
        df = pd.DataFrame({"uid": ["a", "a"], "v": ["p", "q"]})
        out = aggregate_nunique(df, "uid", ["v"])
        assert out["uid_v_ct"].dtype == np.float32
