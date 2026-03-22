# src/config.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')

# ── File names ───────────────────────────────────────────────
TRAIN_FILE      = 'application_train.csv'
TEST_FILE       = 'application_test.csv'
MODEL_FILE      = 'xgb_credit_risk_final.pkl'
SUBMISSION_FILE = 'submission.csv'

# Auxiliary tables — loaded if present, skipped if missing
AUX_FILES = [
    'bureau.csv',
    'bureau_balance.csv',
    'previous_application.csv',
    'installments_payments.csv',
    'credit_card_balance.csv',
    'POS_CASH_balance.csv',
]

# ── Column constants ─────────────────────────────────────────
TARGET    = 'TARGET'
ID_COL    = 'SK_ID_CURR'

# ── Preprocessing ────────────────────────────────────────────
MISSING_THRESHOLD  = 0.6      # drop columns with >60% missing
DAYS_EMPLOYED_ANOM = 365243   # known data entry error value

# ── Train / val split ────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Optuna ───────────────────────────────────────────────────
N_TRIALS  = 100
CV_FOLDS  = 5

# ── XGBoost fixed params (non-tunable) ───────────────────────
XGB_FIXED_PARAMS = {
    'eval_metric'           : 'auc',
    'early_stopping_rounds' : 30,
    'random_state'          : 42,
    'n_jobs'                : -1,
    'tree_method'           : 'hist',
    'device'                : 'cuda',   # change to 'cpu' if no GPU
}

# ── Optuna search space bounds ───────────────────────────────
OPTUNA_SPACE = {
    'n_estimators'      : (300,  1000, 100),   # min, max, step
    'max_depth'         : (3,    9),
    'min_child_weight'  : (1,    10),
    'learning_rate'     : (0.005, 0.2),        # log scale
    'subsample'         : (0.5,  1.0),
    'colsample_bytree'  : (0.4,  1.0),
    'colsample_bylevel' : (0.4,  1.0),
    'reg_alpha'         : (1e-8, 10.0),        # log scale
    'reg_lambda'        : (1e-8, 10.0),        # log scale
    'gamma'             : (0.0,  5.0),
    'min_split_loss'    : (0.0,  2.0),
}

# ── EXT_SOURCE columns ───────────────────────────────────────
EXT_SOURCE_COLS = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']