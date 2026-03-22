# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.config import EXT_SOURCE_COLS, ID_COL, TARGET


def add_application_features(df):
    """Ratio-based features from the main application table."""
    df = df.copy()
    df['CREDIT_INCOME_RATIO']  = df['AMT_CREDIT']  / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_TERM']          = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
    df['INCOME_PER_PERSON']    = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)

    if 'AMT_GOODS_PRICE' in df.columns:
        df['GOODS_PRICE_CREDIT_DIFF'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
        df['GOODS_TO_CREDIT_RATIO']   = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)

    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365

    if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
        df['EMPLOYMENT_TO_AGE_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)

    return df


def add_ext_source_features(df):
    """Combine EXT_SOURCE scores — top predictors in this dataset."""
    cols_present = [c for c in EXT_SOURCE_COLS if c in df.columns]

    if len(cols_present) == 3:
        df['EXT_SOURCE_MEAN'] = df[cols_present].mean(axis=1)
        df['EXT_SOURCE_MIN']  = df[cols_present].min(axis=1)
        df['EXT_SOURCE_MAX']  = df[cols_present].max(axis=1)
        df['EXT_SOURCE_PROD'] = df[cols_present[0]] * df[cols_present[1]] * df[cols_present[2]]
        df['EXT_SOURCE_STD']  = df[cols_present].std(axis=1)

    return df


def encode_categoricals(df):
    """Label encode all object columns. Returns df and fitted encoders dict."""
    encoders = {}
    cat_cols  = df.select_dtypes(include=['object']).columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def apply_encoders(df, encoders):
    """Apply previously fitted encoders to a new dataframe (e.g. test set)."""
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def build_features(df):
    """Run full feature engineering pipeline. Returns df and encoders."""
    print("\nBuilding features...")

    df = add_application_features(df)
    print(f"  After application features : {df.shape}")

    df = add_ext_source_features(df)
    print(f"  After EXT_SOURCE features  : {df.shape}")

    df, encoders = encode_categoricals(df)
    print(f"  After label encoding       : {df.shape}")

    return df, encoders


def get_X_y(df):
    """Split into features and target. Returns X, y."""
    drop_cols = [TARGET, ID_COL]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET] if TARGET in df.columns else None
    return X, y
