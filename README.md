# Credit Risk Prediction — Home Credit Default Risk

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![Optuna](https://img.shields.io/badge/Optuna-3.x-green)
![MLflow](https://img.shields.io/badge/MLflow-3.x-blue)
![Tests](https://img.shields.io/badge/Tests-24%20passed-brightgreen)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.786-yellow)

## Problem

Many people struggle to get loans due to insufficient credit history. Home Credit uses alternative data to predict whether an applicant will repay a loan — helping financial institutions make better lending decisions while reducing default risk.

This project builds an end-to-end credit risk prediction system trained on **307,511 loan applications** across **7 relational tables**, achieving a validation ROC-AUC of **0.786**.

---

## Results

| Metric | Value |
|---|---|
| Validation ROC-AUC | **0.786** |
| Kaggle Public Leaderboard | TBD |
| F1-Optimal Threshold | 0.677 |
| Best Iteration (early stopping) | 698 |
| Training samples | 307,511 |
| Features engineered | 157 |

### Classification Report (F1-optimal threshold 0.677)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Non-default (0) | 0.95 | 0.90 | 0.92 |
| Default (1) | 0.28 | 0.43 | 0.34 |

---

## Key Design Decisions

**Why `scale_pos_weight` instead of SMOTE?**
The dataset has an 11:1 class imbalance. Oversampling with SMOTE on 300k rows inflates training time without consistent AUC gains. `scale_pos_weight` handles imbalance natively in XGBoost with no data duplication.

**Why threshold 0.677 instead of 0.50?**
In credit risk, missing a defaulter (false negative) costs more than a false alarm (false positive). The optimal threshold was found by maximising F1 on the minority class rather than overall accuracy.

**Why Optuna instead of GridSearch?**
Optuna's TPE sampler finds better hyperparameters in fewer trials than grid or random search. 100 trials × 5-fold CV explores the space intelligently, not exhaustively.

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
│   └── test_features.py         # 24 unit tests (pytest)
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

---

## Feature Engineering Highlights

- **EXT_SOURCE combinations** — mean, min, max, product and std of 3 external credit scores (top predictors)
- **Bureau DPD rate** — ratio of months with overdue payments across all past loans
- **Installment payment ratio** — how much of each instalment was actually paid
- **Late payment rate** — proportion of instalments paid after due date
- **Credit utilization** — credit card balance / limit ratio
- **Age and employment ratios** — age in years, employment-to-age ratio
- **Approval rate** — proportion of previous applications that were approved

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place CSVs in data/ folder
Download from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place all CSV files in the `data/` directory.

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

### 5. View MLflow dashboard
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://127.0.0.1:5000

### 6. Run unit tests
```bash
pytest tests/ -v
```

---

## Experiment Tracking (MLflow)

All training runs are tracked with MLflow, logging:
- All 11 Optuna hyperparameters
- Dataset statistics (train size, features, class weight)
- Validation ROC-AUC, optimal threshold, best iteration
- Feature importance CSV
- Full XGBoost model artifact

---

## Tech Stack

| Tool | Purpose |
|---|---|
| XGBoost | Gradient boosted trees (GPU-accelerated) |
| Optuna | Bayesian hyperparameter optimisation |
| scikit-learn | Preprocessing, cross-validation, metrics |
| MLflow | Experiment tracking and model logging |
| pandas / numpy | Data manipulation |
| pytest | Unit testing (24 tests) |
| CUDA / RTX 3050 | GPU-accelerated training |

---

## Dataset

[Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)

## Results

| Metric | Value |
|---|---|
| Validation ROC-AUC | **0.786** |
| Kaggle Private Score | **0.778** |
| Kaggle Public Score | **0.782** |
| F1-Optimal Threshold | 0.677 |
| Best Iteration (early stopping) | 698 |
| Training samples | 307,511 |
| Features engineered | 157 |
