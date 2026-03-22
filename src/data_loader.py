# src/data_loader.py
import os
import pandas as pd
import numpy as np
from src.config import (
    DATA_DIR, TRAIN_FILE, TEST_FILE,
    ID_COL, MISSING_THRESHOLD, DAYS_EMPLOYED_ANOM
)


def load_csv(filename):
    """Load a CSV from DATA_DIR. Returns None if file not found."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  ⚠️  {filename} not found — skipping")
        return None
    df = pd.read_csv(filepath)
    print(f"  ✅ {filename:<45} {str(df.shape):>15}")
    return df


def load_train():
    """Load application_train.csv"""
    df = load_csv(TRAIN_FILE)
    if df is None:
        raise FileNotFoundError(f"{TRAIN_FILE} is required but not found in {DATA_DIR}")
    return df


def load_test():
    """Load application_test.csv"""
    df = load_csv(TEST_FILE)
    if df is None:
        raise FileNotFoundError(f"{TEST_FILE} is required but not found in {DATA_DIR}")
    return df


def load_bureau():
    """Load and merge bureau + bureau_balance. Returns None if bureau.csv missing."""
    bureau = load_csv('bureau.csv')
    if bureau is None:
        return None

    bb = load_csv('bureau_balance.csv')
    if bb is not None:
        bb['DPD_FLAG']    = bb['STATUS'].isin(['1','2','3','4','5']).astype(int)
        bb['CLOSED_FLAG'] = (bb['STATUS'] == 'C').astype(int)

        bb_agg = bb.groupby('SK_ID_BUREAU').agg(
            BB_MONTHS_COUNT     = ('MONTHS_BALANCE', 'count'),
            BB_STATUS_C_COUNT   = ('STATUS',   lambda x: (x == 'C').sum()),
            BB_STATUS_X_COUNT   = ('STATUS',   lambda x: (x == 'X').sum()),
            BB_STATUS_BAD_COUNT = ('DPD_FLAG', 'sum'),
        ).reset_index()
        bb_agg['BB_BAD_RATE'] = (
            bb_agg['BB_STATUS_BAD_COUNT'] / (bb_agg['BB_MONTHS_COUNT'] + 1)
        )
        bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

    bureau_agg = bureau.groupby(ID_COL).agg(
        BUREAU_LOAN_COUNT          = ('SK_ID_BUREAU',        'count'),
        BUREAU_ACTIVE_COUNT        = ('CREDIT_ACTIVE',       lambda x: (x == 'Active').sum()),
        BUREAU_CLOSED_COUNT        = ('CREDIT_ACTIVE',       lambda x: (x == 'Closed').sum()),
        BUREAU_AMT_CREDIT_SUM      = ('AMT_CREDIT_SUM',      'sum'),
        BUREAU_AMT_CREDIT_DEBT_SUM = ('AMT_CREDIT_SUM_DEBT', 'sum'),
        BUREAU_AMT_CREDIT_OVERDUE  = ('AMT_CREDIT_SUM_OVERDUE', 'sum'),
        BUREAU_DAYS_CREDIT_MEAN    = ('DAYS_CREDIT',         'mean'),
        BUREAU_DAYS_CREDIT_MAX     = ('DAYS_CREDIT',         'max'),
        BUREAU_DAYS_ENDDATE_MEAN   = ('DAYS_CREDIT_ENDDATE', 'mean'),
        BUREAU_CNT_PROLONG_SUM     = ('CNT_CREDIT_PROLONG',  'sum'),
        BB_MONTHS_COUNT_MEAN       = ('BB_MONTHS_COUNT',     'mean'),
        BB_BAD_COUNT_MEAN          = ('BB_STATUS_BAD_COUNT', 'mean'),
        BB_BAD_COUNT_SUM           = ('BB_STATUS_BAD_COUNT', 'sum'),
        BB_BAD_RATE_MEAN           = ('BB_BAD_RATE',         'mean'),
        BB_BAD_RATE_MAX            = ('BB_BAD_RATE',         'max'),
    ).reset_index()

    bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = (
        bureau_agg['BUREAU_AMT_CREDIT_DEBT_SUM'] /
        (bureau_agg['BUREAU_AMT_CREDIT_SUM'] + 1)
    )
    bureau_agg['BUREAU_ACTIVE_RATIO'] = (
        bureau_agg['BUREAU_ACTIVE_COUNT'] /
        (bureau_agg['BUREAU_LOAN_COUNT'] + 1)
    )
    return bureau_agg


def load_pos_cash():
    """Load and aggregate POS_CASH_balance.csv"""
    pos = load_csv('POS_CASH_balance.csv')
    if pos is None:
        return None

    return pos.groupby(ID_COL).agg(
        POS_MONTHS_BALANCE_MEAN = ('MONTHS_BALANCE',       'mean'),
        POS_CNT_INSTALMENT_MEAN = ('CNT_INSTALMENT',       'mean'),
        POS_SK_DPD_MAX          = ('SK_DPD',               'max'),
        POS_SK_DPD_DEF_SUM      = ('SK_DPD_DEF',           'sum'),
        POS_COMPLETED_COUNT     = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Completed').sum()),
    ).reset_index()


def load_previous_application():
    """Load and aggregate previous_application.csv"""
    prev = load_csv('previous_application.csv')
    if prev is None:
        return None

    prev_agg = prev.groupby(ID_COL).agg(
        PREV_APP_COUNT              = ('SK_ID_PREV',           'count'),
        PREV_APPROVED_COUNT         = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
        PREV_REFUSED_COUNT          = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
        PREV_AMT_CREDIT_MEAN        = ('AMT_CREDIT',           'mean'),
        PREV_AMT_ANNUITY_MEAN       = ('AMT_ANNUITY',          'mean'),
        PREV_AMT_DOWN_PAYMENT_MEAN  = ('AMT_DOWN_PAYMENT',     'mean'),
        PREV_DAYS_DECISION_MEAN     = ('DAYS_DECISION',        'mean'),
        PREV_DAYS_DECISION_MIN      = ('DAYS_DECISION',        'min'),
        PREV_RATE_DOWN_PAYMENT_MEAN = ('RATE_DOWN_PAYMENT',    'mean'),
    ).reset_index()

    prev_agg['PREV_APPROVAL_RATE'] = (
        prev_agg['PREV_APPROVED_COUNT'] / (prev_agg['PREV_APP_COUNT'] + 1)
    )
    return prev_agg


def load_installments():
    """Load and aggregate installments_payments.csv"""
    inst = load_csv('installments_payments.csv')
    if inst is None:
        return None

    inst['DPD'] = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)
    inst['DBD'] = (inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']).clip(lower=0)
    inst['PAYMENT_DIFF']  = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    inst['PAYMENT_RATIO'] = inst['AMT_PAYMENT'] / (inst['AMT_INSTALMENT'] + 1)

    inst_agg = inst.groupby(ID_COL).agg(
        INST_COUNT              = ('SK_ID_PREV',    'count'),
        INST_AMT_PAYMENT_SUM    = ('AMT_PAYMENT',   'sum'),
        INST_AMT_PAYMENT_MEAN   = ('AMT_PAYMENT',   'mean'),
        INST_PAYMENT_DIFF_MEAN  = ('PAYMENT_DIFF',  'mean'),
        INST_PAYMENT_DIFF_MAX   = ('PAYMENT_DIFF',  'max'),
        INST_PAYMENT_RATIO_MEAN = ('PAYMENT_RATIO', 'mean'),
        INST_DPD_MEAN           = ('DPD',           'mean'),
        INST_DPD_MAX            = ('DPD',           'max'),
        INST_DPD_SUM            = ('DPD',           'sum'),
        INST_DBD_MEAN           = ('DBD',           'mean'),
        INST_NUM_LATE           = ('DPD',           lambda x: (x > 0).sum()),
    ).reset_index()

    inst_agg['INST_LATE_PAYMENT_RATE'] = (
        inst_agg['INST_NUM_LATE'] / (inst_agg['INST_COUNT'] + 1)
    )
    return inst_agg


def load_credit_card():
    """Load and aggregate credit_card_balance.csv"""
    cc = load_csv('credit_card_balance.csv')
    if cc is None:
        return None

    cc_agg = cc.groupby(ID_COL).agg(
        CC_COUNT                 = ('SK_ID_PREV',              'count'),
        CC_AMT_BALANCE_MEAN      = ('AMT_BALANCE',             'mean'),
        CC_AMT_BALANCE_MAX       = ('AMT_BALANCE',             'max'),
        CC_AMT_CREDIT_LIMIT_MEAN = ('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
        CC_CNT_DRAWINGS_MEAN     = ('CNT_DRAWINGS_CURRENT',    'mean'),
        CC_CNT_DRAWINGS_MAX      = ('CNT_DRAWINGS_CURRENT',    'max'),
        CC_SK_DPD_MEAN           = ('SK_DPD',                  'mean'),
        CC_SK_DPD_MAX            = ('SK_DPD',                  'max'),
        CC_AMT_PAYMENT_MEAN      = ('AMT_PAYMENT_CURRENT',     'mean'),
    ).reset_index()

    cc_agg['CC_UTILIZATION'] = (
        cc_agg['CC_AMT_BALANCE_MEAN'] /
        (cc_agg['CC_AMT_CREDIT_LIMIT_MEAN'] + 1)
    )
    return cc_agg


def build_dataset(df):
    """Merge all auxiliary tables into main dataframe."""
    print("\nMerging auxiliary tables...")

    for loader, name in [
        (load_bureau,               'Bureau + Balance'),
        (load_pos_cash,             'POS Cash'),
        (load_previous_application, 'Previous Application'),
        (load_installments,         'Installments'),
        (load_credit_card,          'Credit Card'),
    ]:
        agg = loader()
        if agg is not None:
            df = df.merge(agg, on=ID_COL, how='left')
            print(f"  After {name:<25} shape: {df.shape}")

    return df


def clean(df):
    """Drop high-missing columns, fix anomalies, fill NaNs."""
    missing_ratio = df.isnull().mean()
    cols_to_drop  = missing_ratio[missing_ratio > MISSING_THRESHOLD].index
    df = df.drop(columns=cols_to_drop)

    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == DAYS_EMPLOYED_ANOM).astype(int)
        df['DAYS_EMPLOYED']      = df['DAYS_EMPLOYED'].replace(DAYS_EMPLOYED_ANOM, np.nan)

    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df
