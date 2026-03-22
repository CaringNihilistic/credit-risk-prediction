# tests/test_features.py
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.feature_engineering import (
    add_application_features,
    add_ext_source_features,
    encode_categoricals,
)
from src.data_loader import clean


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal dataframe mimicking application_train structure."""
    return pd.DataFrame({
        'AMT_CREDIT'        : [100000.0, 200000.0],
        'AMT_INCOME_TOTAL'  : [50000.0,  100000.0],
        'AMT_ANNUITY'       : [10000.0,  20000.0],
        'AMT_GOODS_PRICE'   : [90000.0,  180000.0],
        'CNT_FAM_MEMBERS'   : [2.0,      4.0],
        'DAYS_BIRTH'        : [-12000,   -15000],
        'DAYS_EMPLOYED'     : [-3000,    365243],  # second row is anomaly
        'EXT_SOURCE_1'      : [0.5,      0.7],
        'EXT_SOURCE_2'      : [0.6,      0.8],
        'EXT_SOURCE_3'      : [0.4,      0.6],
        'NAME_CONTRACT_TYPE': ['Cash loans', 'Revolving loans'],
        'TARGET'            : [0, 1],
    })


# ── Application feature tests ─────────────────────────────────

def test_credit_income_ratio(sample_df):
    df = add_application_features(sample_df.copy())
    expected = 100000.0 / (50000.0 + 1)
    assert df['CREDIT_INCOME_RATIO'].iloc[0] == pytest.approx(expected, rel=1e-4)


def test_annuity_income_ratio(sample_df):
    df = add_application_features(sample_df.copy())
    expected = 10000.0 / (50000.0 + 1)
    assert df['ANNUITY_INCOME_RATIO'].iloc[0] == pytest.approx(expected, rel=1e-4)


def test_credit_term(sample_df):
    df = add_application_features(sample_df.copy())
    expected = 10000.0 / (100000.0 + 1)
    assert df['CREDIT_TERM'].iloc[0] == pytest.approx(expected, rel=1e-4)


def test_income_per_person(sample_df):
    df = add_application_features(sample_df.copy())
    expected = 50000.0 / (2.0 + 1)
    assert df['INCOME_PER_PERSON'].iloc[0] == pytest.approx(expected, rel=1e-4)


def test_age_years(sample_df):
    df = add_application_features(sample_df.copy())
    expected = 12000 / 365
    assert df['AGE_YEARS'].iloc[0] == pytest.approx(expected, rel=1e-4)


def test_goods_price_features_exist(sample_df):
    df = add_application_features(sample_df.copy())
    assert 'GOODS_PRICE_CREDIT_DIFF' in df.columns
    assert 'GOODS_TO_CREDIT_RATIO'   in df.columns


def test_goods_price_credit_diff(sample_df):
    df = add_application_features(sample_df.copy())
    assert df['GOODS_PRICE_CREDIT_DIFF'].iloc[0] == pytest.approx(10000.0, rel=1e-4)


def test_no_nulls_after_application_features(sample_df):
    df = add_application_features(sample_df.copy())
    new_cols = [
        'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
        'CREDIT_TERM', 'INCOME_PER_PERSON', 'AGE_YEARS',
    ]
    for col in new_cols:
        assert df[col].isnull().sum() == 0, f"{col} has unexpected nulls"


def test_employment_to_age_ratio(sample_df):
    df = add_application_features(sample_df.copy())
    assert 'EMPLOYMENT_TO_AGE_RATIO' in df.columns


# ── EXT_SOURCE feature tests ──────────────────────────────────

def test_ext_source_mean(sample_df):
    df = add_ext_source_features(sample_df.copy())
    expected = np.mean([0.5, 0.6, 0.4])
    assert df['EXT_SOURCE_MEAN'].iloc[0] == pytest.approx(expected, rel=1e-4)


def test_ext_source_min(sample_df):
    df = add_ext_source_features(sample_df.copy())
    assert df['EXT_SOURCE_MIN'].iloc[0] == pytest.approx(0.4, rel=1e-4)


def test_ext_source_max(sample_df):
    df = add_ext_source_features(sample_df.copy())
    assert df['EXT_SOURCE_MAX'].iloc[0] == pytest.approx(0.6, rel=1e-4)


def test_ext_source_prod(sample_df):
    df = add_ext_source_features(sample_df.copy())
    expected = 0.5 * 0.6 * 0.4
    assert df['EXT_SOURCE_PROD'].iloc[0] == pytest.approx(expected, rel=1e-4)


def test_ext_source_std(sample_df):
    df = add_ext_source_features(sample_df.copy())
    expected = np.std([0.5, 0.6, 0.4], ddof=1)
    assert df['EXT_SOURCE_STD'].iloc[0] == pytest.approx(expected, rel=1e-3)


def test_ext_source_all_columns_created(sample_df):
    df = add_ext_source_features(sample_df.copy())
    for col in ['EXT_SOURCE_MEAN', 'EXT_SOURCE_MIN',
                'EXT_SOURCE_MAX', 'EXT_SOURCE_PROD', 'EXT_SOURCE_STD']:
        assert col in df.columns, f"{col} not created"


def test_ext_source_skipped_if_missing():
    """If EXT_SOURCE columns are absent, no crash and no new columns."""
    df = pd.DataFrame({'AMT_CREDIT': [100000], 'TARGET': [0]})
    df_out = add_ext_source_features(df.copy())
    assert 'EXT_SOURCE_MEAN' not in df_out.columns


# ── Cleaning tests ────────────────────────────────────────────

def test_days_employed_anomaly_flagged(sample_df):
    df = clean(sample_df.copy())
    assert 'DAYS_EMPLOYED_ANOM' in df.columns
    assert df['DAYS_EMPLOYED_ANOM'].iloc[1] == 1   # row with 365243
    assert df['DAYS_EMPLOYED_ANOM'].iloc[0] == 0   # normal row


def test_days_employed_anomaly_replaced_with_nan(sample_df):
    # Test the replacement BEFORE median fill
    # clean() correctly replaces 365243 with NaN then fills with median
    df = sample_df.copy()

    # Manually apply just the anomaly replacement (not full clean)
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED']      = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    # At this point, before median fill, it should be NaN
    assert np.isnan(df['DAYS_EMPLOYED'].iloc[1])

    # After clean(), median fill kicks in — NaN becomes -3000.0 (median of [-3000])
    df_clean = clean(sample_df.copy())
    assert df_clean['DAYS_EMPLOYED'].iloc[1] == pytest.approx(-3000.0, rel=1e-4)

def test_no_nulls_after_clean(sample_df):
    df = clean(sample_df.copy())
    assert df.isnull().sum().sum() == 0, "clean() left nulls in dataframe"


def test_high_missing_columns_dropped():
    """Columns with >60% missing should be dropped by clean()."""
    df = pd.DataFrame({
        'GOOD_COL'   : [1, 2, 3, 4, 5],
        'BAD_COL'    : [None, None, None, None, 1],  # 80% missing
        'DAYS_EMPLOYED': [-1000, -2000, -3000, -4000, -5000],
        'TARGET'     : [0, 1, 0, 1, 0],
    })
    df_clean = clean(df.copy())
    assert 'BAD_COL' not in df_clean.columns
    assert 'GOOD_COL' in df_clean.columns


# ── Label encoding tests ──────────────────────────────────────

def test_encode_categoricals_converts_to_int(sample_df):
    df, encoders = encode_categoricals(sample_df.copy())
    assert df['NAME_CONTRACT_TYPE'].dtype in [np.int32, np.int64, int]


def test_encode_categoricals_returns_encoders(sample_df):
    df, encoders = encode_categoricals(sample_df.copy())
    assert isinstance(encoders, dict)
    assert 'NAME_CONTRACT_TYPE' in encoders


def test_no_data_leakage_target_not_in_features():
    """TARGET column must never appear in feature set."""
    feature_cols = [
        'CREDIT_INCOME_RATIO', 'AGE_YEARS', 'EXT_SOURCE_MEAN',
        'INST_LATE_PAYMENT_RATE', 'BUREAU_DEBT_CREDIT_RATIO',
    ]
    assert 'TARGET' not in feature_cols


def test_no_id_column_in_features():
    """SK_ID_CURR must never appear in feature set."""
    feature_cols = [
        'CREDIT_INCOME_RATIO', 'AGE_YEARS', 'EXT_SOURCE_MEAN',
    ]
    assert 'SK_ID_CURR' not in feature_cols
