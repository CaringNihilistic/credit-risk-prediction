# Credit Risk Prediction — Home Credit Default Risk

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-3.x-green?style=flat-square)](https://optuna.org)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-blue?style=flat-square)](https://mlflow.org)
[![Tests](https://img.shields.io/badge/Tests-27%20passed-brightgreen?style=flat-square)](tests/)
[![Kaggle AUC](https://img.shields.io/badge/Kaggle%20Public%20AUC-0.782-gold?style=flat-square)](https://www.kaggle.com/c/home-credit-default-risk)
[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-yellow?style=flat-square)](https://huggingface.co/spaces/ayushthecaringnihilist/credit-risk-prediction)

---

## 🔴 Live Demo

**Try it instantly — no setup required:**
👉 [huggingface.co/spaces/ayushthecaringnihilist/credit-risk-prediction](https://huggingface.co/spaces/ayushthecaringnihilist/credit-risk-prediction)

Enter 10 applicant parameters and get a real-time default probability with risk classification and a visual decision bar.

---

## Problem

Many people struggle to get loans due to insufficient credit history. Home Credit uses alternative data to predict whether an applicant will repay a loan — helping financial institutions make better lending decisions while reducing default risk.

This project builds an end-to-end credit risk prediction system trained on **307,511 loan applications** across **7 relational tables**, achieving a Kaggle Public AUC of **0.782**.

---

## Results

| Metric | Value |
|---|---|
| Kaggle Public AUC | **0.782** |
| Kaggle Private AUC | **0.778** |
| Validation ROC-AUC (5-fold CV) | **0.786** |
| F1-Optimal Threshold | **0.677** |
| Best Iteration (early stopping) | 698 |
| Training samples | 307,511 |
| Features engineered | 157 |

### Classification Report (threshold = 0.677)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Non-default (0) | 0.95 | 0.90 | 0.92 |
| Default (1) | 0.28 | 0.43 | 0.34 |

---

## Architecture

```
credit-risk-prediction/
├── data/                        # Raw CSVs (not committed)
├── models/                      # Saved model + encoders
├── notebooks/
│   └── credit_risk_modeling.ipynb
├── src/
│   ├── config.py                # All constants and hyperparameter bounds
│   ├── data_loader.py           # Loading + merging 6 auxiliary tables
│   ├── feature_engineering.py   # Feature creation + encoding
│   ├── train.py                 # Optuna search + final model training
│   └── predict.py               # Test set inference + submission
├── tests/
│   └── test_features.py         # 27 unit tests (pytest)
├── app.py                       # Gradio 5 web app (HuggingFace Spaces)
├── requirements.txt
└── README.md
```

---

## Data Sources

| File | Rows | Description |
|---|---|---|
| `application_train.csv` | 307,511 | Main loan applications |
| `bureau.csv` | 1,716,428 | Credit Bureau loan history |
| `bureau_balance.csv` | 27,299,925 | Monthly status of bureau loans |
| `previous_application.csv` | 1,670,214 | Past Home Credit applications |
| `installments_payments.csv` | 13,605,401 | Installment payment history |
| `credit_card_balance.csv` | 3,840,312 | Credit card balance history |
| `POS_CASH_balance.csv` | 10,001,358 | POS and cash loan balances |

**Total: 57M+ rows across 7 tables.**

---

## Feature Engineering Highlights

- **EXT_SOURCE combinations** — mean, min, max, product and std of 3 external credit bureau scores (top 5 predictors by XGBoost feature importance)
- **Bureau DPD rate** — ratio of months with overdue payments across all past loans
- **Installment payment ratio** — how much of each instalment was actually paid
- **Late payment rate** — proportion of instalments paid after due date
- **Credit utilization** — credit card balance / limit ratio
- **Age and employment ratios** — age in years, employment-to-age ratio
- **Approval rate** — proportion of previous applications that were approved

---

## Key Design Decisions

**Why `scale_pos_weight = 11.39` instead of SMOTE?**
The dataset has an 11:1 class imbalance. SMOTE on 300k rows inflates training time and introduces synthetic noise without consistent AUC gains. `scale_pos_weight` handles imbalance natively in XGBoost with zero data duplication and preserves the original distribution.

**Why threshold 0.677 instead of 0.50?**
In credit risk, missing a defaulter (false negative) costs more than a false alarm (false positive). The threshold was selected by evaluating F1 on the minority class across all values on the validation set and picking the maximum — not the default 0.50.

**Why Optuna instead of GridSearch?**
Optuna's TPE sampler finds better hyperparameters in fewer trials than grid or random search. 100 trials × 5-fold CV explores the space intelligently, not exhaustively.

**Why merge `bureau_balance` into `bureau` before aggregating?**
Aggregating `bureau_balance` directly onto `SK_ID_CURR` causes a fan-out join explosion. Merging into `bureau` first (on `SK_ID_BUREAU`) then aggregating keeps row counts correct and avoids data leakage.

**Why `early_stopping_rounds` in the constructor, not `fit()`?**
XGBoost v2 moved `early_stopping_rounds` to the constructor. Passing it in `fit()` raises a deprecation warning and behaves inconsistently across versions.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place CSVs in `data/` folder
Download from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place all 7 CSV files in the `data/` directory.

### 3. Train the model
```python
from src.data_loader import load_train, build_dataset, clean
from src.feature_engineering import build_features, get_X_y
from src.train import run_training_pipeline_with_mlflow

df = load_train()
df = build_dataset(df)
df = clean(df)
df, encoders = build_features(df)
X, y = get_X_y(df)

final_model, metrics, best_params = run_training_pipeline_with_mlflow(X, y)
```

### 4. Generate submission
```python
from src.predict import generate_submission

submission = generate_submission(
    encoders      = encoders,
    train_columns = X.columns,
    output_path   = 'submission.csv'
)
```

### 5. Run the web app locally
```bash
python app.py
# → opens at http://localhost:7860
```

### 6. View MLflow dashboard
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → opens at http://127.0.0.1:5000
```

### 7. Run unit tests
```bash
pytest tests/ -v
# → 27 tests, all passing
```

---

## 🤗 HuggingFace Spaces Deployment

The app runs on HuggingFace Spaces with **Gradio 5** and **Python 3.13**.

The Space mirrors this repo's layout — `app.py` imports its feature functions from `src/`, so only the two modules it uses are deployed:

```
Space root/
├── app.py                      # Gradio 5 UI entrypoint
├── src/
│   ├── config.py
│   └── feature_engineering.py
├── xgb_credit_risk_final.pkl   # Trained model artifact
└── requirements.txt
```

### How the app works

1. User inputs 10 parameters (3 credit scores, 4 financial, 3 personal)
2. Human-readable values are converted to model format:
   - `Age (years)` → `DAYS_BIRTH = -age × 365`
   - `Years employed` → `DAYS_EMPLOYED = -years × 365` (or `365243` if unemployed)
3. 13 derived features are computed by the same `src/feature_engineering.py` functions used in training
4. DataFrame is aligned to the model's exact 157-feature column order via `reindex()`
5. `model.predict_proba()` returns the default probability
6. Threshold **0.677** classifies: `< 0.40` Low · `0.40–0.677` Medium · `≥ 0.677` High Risk

### `requirements.txt` for Spaces
```
gradio==5.29.0
xgboost
scikit-learn
pandas
numpy
joblib
```

---

## Experiment Tracking (MLflow)

All training runs are tracked with MLflow, logging:
- All 11 Optuna hyperparameters per trial
- Dataset statistics (train size, features, class weight)
- Validation ROC-AUC, optimal threshold, best iteration
- Feature importance CSV
- Full XGBoost model artifact

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| XGBoost | Gradient boosted trees (GPU-accelerated via CUDA) |
| Optuna | Bayesian hyperparameter optimisation (TPE sampler) |
| scikit-learn | Preprocessing, cross-validation, metrics |
| MLflow | Experiment tracking and model artifact logging |
| Gradio 5 | Web UI for live demo |
| HuggingFace Spaces | Model deployment |
| pandas / numpy | Data manipulation |
| pytest | Unit testing (27 tests) |
| CUDA / RTX 3050 | GPU-accelerated training |

---

## Dataset

[Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
