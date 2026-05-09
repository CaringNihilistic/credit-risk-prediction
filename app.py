# app.py
import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# ── Load model ───────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xgb_credit_risk_final.pkl")
model = joblib.load(MODEL_PATH)

# ── Top 10 most important features for the demo UI ───────────
# These are the top features from your feature importance output
FEATURE_DEFAULTS = {
    "EXT_SOURCE_1"        : 0.50,
    "EXT_SOURCE_2"        : 0.55,
    "EXT_SOURCE_3"        : 0.60,
    "AMT_CREDIT"          : 500000.0,
    "AMT_INCOME_TOTAL"    : 150000.0,
    "AMT_ANNUITY"         : 25000.0,
    "DAYS_BIRTH"          : -12000,
    "DAYS_EMPLOYED"       : -2000,
    "AMT_GOODS_PRICE"     : 450000.0,
    "CNT_FAM_MEMBERS"     : 2.0,
}

def predict(ext1, ext2, ext3, amt_credit, amt_income,
            amt_annuity, days_birth, days_employed,
            amt_goods, cnt_family):

    # Build feature dict with submitted values
    data = {
        "EXT_SOURCE_1"     : ext1,
        "EXT_SOURCE_2"     : ext2,
        "EXT_SOURCE_3"     : ext3,
        "AMT_CREDIT"       : amt_credit,
        "AMT_INCOME_TOTAL" : amt_income,
        "AMT_ANNUITY"      : amt_annuity,
        "DAYS_BIRTH"       : days_birth,
        "DAYS_EMPLOYED"    : days_employed,
        "AMT_GOODS_PRICE"  : amt_goods,
        "CNT_FAM_MEMBERS"  : cnt_family,
    }

    # Derived features (same as training pipeline)
    data["CREDIT_INCOME_RATIO"]  = data["AMT_CREDIT"] / (data["AMT_INCOME_TOTAL"] + 1)
    data["ANNUITY_INCOME_RATIO"] = data["AMT_ANNUITY"] / (data["AMT_INCOME_TOTAL"] + 1)
    data["CREDIT_TERM"]          = data["AMT_ANNUITY"] / (data["AMT_CREDIT"] + 1)
    data["INCOME_PER_PERSON"]    = data["AMT_INCOME_TOTAL"] / (data["CNT_FAM_MEMBERS"] + 1)
    data["GOODS_PRICE_CREDIT_DIFF"] = data["AMT_CREDIT"] - data["AMT_GOODS_PRICE"]
    data["GOODS_TO_CREDIT_RATIO"]   = data["AMT_GOODS_PRICE"] / (data["AMT_CREDIT"] + 1)
    data["AGE_YEARS"]               = -data["DAYS_BIRTH"] / 365
    data["EMPLOYMENT_TO_AGE_RATIO"] = data["DAYS_EMPLOYED"] / (data["DAYS_BIRTH"] + 1)
    data["EXT_SOURCE_MEAN"] = np.mean([ext1, ext2, ext3])
    data["EXT_SOURCE_MIN"]  = np.min([ext1, ext2, ext3])
    data["EXT_SOURCE_MAX"]  = np.max([ext1, ext2, ext3])
    data["EXT_SOURCE_PROD"] = ext1 * ext2 * ext3
    data["EXT_SOURCE_STD"]  = np.std([ext1, ext2, ext3])

    # Build DataFrame aligned to model's expected columns
    df = pd.DataFrame([data])

    # Get model feature names and align
    model_features = model.get_booster().feature_names
    df = df.reindex(columns=model_features, fill_value=0)

    # Predict
    proba = model.predict_proba(df)[0, 1]
    threshold = 0.677  # your optimal threshold

    if proba >= threshold:
        risk_level = "🔴 HIGH RISK — Likely to Default"
        color = "red"
    elif proba >= 0.40:
        risk_level = "🟡 MEDIUM RISK — Monitor Closely"
        color = "orange"
    else:
        risk_level = "🟢 LOW RISK — Likely to Repay"
        color = "green"

    return (
        f"{proba:.2%}",
        risk_level,
        f"Optimal threshold used: {threshold} | "
        f"Model: XGBoost | Kaggle AUC: 0.782"
    )


# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="Credit Risk Prediction", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🏦 Credit Risk Prediction System
    **Home Credit Default Risk** — Predicts the probability of loan default
    using an XGBoost model trained on 307,511 applicants across 7 relational tables.

    > Kaggle Public Leaderboard: **ROC-AUC 0.782** | Features engineered: **157**
    > Built by [Ayush Yadav](https://github.com/CaringNihilistic)
    """)

    gr.Markdown("### 📋 Enter Applicant Details")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**External Credit Scores** *(0 = high risk, 1 = low risk)*")
            ext1 = gr.Slider(0.0, 1.0, value=0.50, step=0.01, label="EXT_SOURCE_1")
            ext2 = gr.Slider(0.0, 1.0, value=0.55, step=0.01, label="EXT_SOURCE_2")
            ext3 = gr.Slider(0.0, 1.0, value=0.60, step=0.01, label="EXT_SOURCE_3")

        with gr.Column():
            gr.Markdown("**Financial Details**")
            amt_credit  = gr.Number(value=500000, label="Loan Amount (AMT_CREDIT)")
            amt_income  = gr.Number(value=150000, label="Annual Income (AMT_INCOME_TOTAL)")
            amt_annuity = gr.Number(value=25000,  label="Annual Annuity (AMT_ANNUITY)")

        with gr.Column():
            gr.Markdown("**Personal Details**")
            days_birth    = gr.Number(value=-12000, label="Days Since Birth (negative)")
            days_employed = gr.Number(value=-2000,  label="Days Employed (negative)")
            amt_goods     = gr.Number(value=450000, label="Goods Price (AMT_GOODS_PRICE)")
            cnt_family    = gr.Number(value=2,      label="Family Members")

    predict_btn = gr.Button("🔍 Predict Default Risk", variant="primary", size="lg")

    gr.Markdown("### 📊 Prediction Result")
    with gr.Row():
        out_proba  = gr.Textbox(label="Default Probability", scale=1)
        out_risk   = gr.Textbox(label="Risk Level", scale=2)
        out_info   = gr.Textbox(label="Model Info", scale=2)

    predict_btn.click(
        fn      = predict,
        inputs  = [ext1, ext2, ext3, amt_credit, amt_income,
                   amt_annuity, days_birth, days_employed,
                   amt_goods, cnt_family],
        outputs = [out_proba, out_risk, out_info]
    )

    gr.Markdown("""
    ---
    ### 📌 About This Model
    | Metric | Value |
    |---|---|
    | Validation ROC-AUC | 0.786 |
    | Kaggle Public Score | 0.782 |
    | Kaggle Private Score | 0.778 |
    | Optimal Threshold | 0.677 |
    | Training Samples | 307,511 |
    | Features Engineered | 157 |

    **Tech Stack:** XGBoost · Optuna · MLflow · scikit-learn · CUDA · pytest
    [GitHub Repository](https://github.com/CaringNihilistic/credit-risk-prediction)
    """)

demo.launch()