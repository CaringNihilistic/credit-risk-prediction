# src/predict.py
import os
import pandas as pd
import numpy as np
from src.config import DATA_DIR, TEST_FILE, ID_COL, TARGET
from src.data_loader import load_test, build_dataset, clean
from src.feature_engineering import build_features, apply_encoders, get_X_y
from src.train import load_model


def generate_submission(encoders, train_columns, output_path='submission.csv'):
    """
    Full prediction pipeline for test set.

    Args:
        encoders      : dict of fitted LabelEncoders from training
        train_columns : X_train.columns — ensures test has same features
        output_path   : where to save submission.csv

    Returns:
        submission DataFrame
    """
    print("=" * 55)
    print("PREDICTION PIPELINE")
    print("=" * 55)

    # ── Load test set ──────────────────────────────────────
    print("\nLoading test data...")
    df_test  = load_test()
    test_ids = df_test[ID_COL].copy()

    # ── Merge auxiliary tables ─────────────────────────────
    df_test = build_dataset(df_test)

    # ── Clean ──────────────────────────────────────────────
    df_test = clean(df_test)

    # ── Feature engineering ────────────────────────────────
    print("\nBuilding features...")
    df_test, _ = build_features(df_test)

    # ── Align columns exactly to training ──────────────────
    # Drop TARGET and ID if present, then reindex to match train
    X_test = df_test.drop(columns=[TARGET, ID_COL], errors='ignore')
    X_test = X_test.reindex(columns=train_columns, fill_value=0)
    print(f"  Test features aligned : {X_test.shape}")

    # ── Load model and predict ─────────────────────────────
    print("\nLoading model and predicting...")
    model      = load_model()
    test_proba = model.predict_proba(X_test)[:, 1]

    # ── Build submission ───────────────────────────────────
    submission = pd.DataFrame({
        ID_COL : test_ids,
        TARGET  : test_proba,
    })

    submission.to_csv(output_path, index=False)

    print(f"\n  Submission saved  -> {output_path}")
    print(f"  Rows              : {len(submission)}")
    print(f"  Score range       : {test_proba.min():.4f} – {test_proba.max():.4f}")
    print(f"  Mean probability  : {test_proba.mean():.4f}")
    print("=" * 55)

    return submission


def predict_single(applicant_data: dict, encoders, train_columns):
    """
    Predict default probability for a single applicant.

    Args:
        applicant_data : dict of feature name -> value
        encoders       : fitted LabelEncoders from training
        train_columns  : X_train.columns

    Returns:
        float — probability of default (0 to 1)

    Example:
        prob = predict_single({
            'AMT_CREDIT'      : 500000,
            'AMT_INCOME_TOTAL': 150000,
            'DAYS_BIRTH'      : -12000,
            'EXT_SOURCE_2'    : 0.65,
        }, encoders, X_train.columns)
        print(f"Default probability: {prob:.2%}")
    """
    model = load_model()

    # Build single-row DataFrame
    df_single = pd.DataFrame([applicant_data])

    # Encode any categorical columns
    df_single = apply_encoders(df_single, encoders)

    # Align to training columns
    df_single = df_single.reindex(columns=train_columns, fill_value=0)

    prob = model.predict_proba(df_single)[0, 1]
    return prob
