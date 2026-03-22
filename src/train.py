# src/train.py
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from xgboost import XGBClassifier
import optuna
import mlflow
import mlflow.xgboost

from src.config import (
    MODEL_DIR, MODEL_FILE,
    TARGET, ID_COL,
    TEST_SIZE, RANDOM_STATE,
    N_TRIALS, CV_FOLDS,
    XGB_FIXED_PARAMS, OPTUNA_SPACE,
)




def split_data(X, y):
    """Stratified train/val split."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    print(f"  Train : {X_train.shape} | Val : {X_val.shape}")
    print(f"  scale_pos_weight = {scale_pos_weight:.2f}")
    return X_train, X_val, y_train, y_val, scale_pos_weight


def build_objective(X_train, y_train, scale_pos_weight):
    """Returns Optuna objective function closed over training data."""

    def objective(trial):
        sp = OPTUNA_SPACE
        n_estimators = trial.suggest_int('n_estimators', sp['n_estimators'][0], sp['n_estimators'][1], step=sp['n_estimators'][2])
        max_depth         = trial.suggest_int  ('max_depth',         *sp['max_depth'])
        min_child_weight  = trial.suggest_int  ('min_child_weight',  *sp['min_child_weight'])
        learning_rate     = trial.suggest_float('learning_rate',     *sp['learning_rate'],     log=True)
        subsample         = trial.suggest_float('subsample',         *sp['subsample'])
        colsample_bytree  = trial.suggest_float('colsample_bytree',  *sp['colsample_bytree'])
        colsample_bylevel = trial.suggest_float('colsample_bylevel', *sp['colsample_bylevel'])
        reg_alpha         = trial.suggest_float('reg_alpha',         *sp['reg_alpha'],         log=True)
        reg_lambda        = trial.suggest_float('reg_lambda',        *sp['reg_lambda'],        log=True)
        gamma             = trial.suggest_float('gamma',             *sp['gamma'])
        min_split_loss    = trial.suggest_float('min_split_loss',    *sp['min_split_loss'])

        cv         = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        auc_scores = []

        for train_idx, valid_idx in cv.split(X_train, y_train):
            Xtr = X_train.iloc[train_idx]
            Xvl = X_train.iloc[valid_idx]
            ytr = y_train.iloc[train_idx]
            yvl = y_train.iloc[valid_idx]

            model = XGBClassifier(
                n_estimators          = n_estimators,
                max_depth             = max_depth,
                min_child_weight      = min_child_weight,
                learning_rate         = learning_rate,
                subsample             = subsample,
                colsample_bytree      = colsample_bytree,
                colsample_bylevel     = colsample_bylevel,
                reg_alpha             = reg_alpha,
                reg_lambda            = reg_lambda,
                gamma                 = gamma,
                min_split_loss        = min_split_loss,
                scale_pos_weight      = scale_pos_weight,
                **XGB_FIXED_PARAMS,
            )
            model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
            auc_scores.append(roc_auc_score(yvl, model.predict_proba(Xvl)[:, 1]))

        return float(np.mean(auc_scores))

    return objective


def run_optuna(X_train, y_train, scale_pos_weight):
    """Run Optuna hyperparameter search. Returns best params dict."""
    print(f"\nRunning Optuna: {N_TRIALS} trials x {CV_FOLDS}-fold CV ...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def print_callback(study, trial):
        if trial.number % 10 == 0:
            print(f"  Trial {trial.number:>3}/{N_TRIALS} | "
                  f"Current: {trial.value:.6f} | "
                  f"Best: {study.best_value:.6f}")

    study = optuna.create_study(
        direction = 'maximize',
        sampler   = optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        build_objective(X_train, y_train, scale_pos_weight),
        n_trials          = N_TRIALS,
        callbacks         = [print_callback],
        show_progress_bar = False,
    )

    print(f"\n  Best CV ROC-AUC : {study.best_value:.6f}")
    print(f"  Best params     : {study.best_params}")
    return study.best_params, study


def train_final_model(X_train, y_train, X_val, y_val, best_params, scale_pos_weight):
    """Train final XGBoost model with best Optuna params."""
    print("\nTraining final model ...")

    final_model = XGBClassifier(
        n_estimators          = best_params['n_estimators'],
        max_depth             = best_params['max_depth'],
        min_child_weight      = best_params['min_child_weight'],
        learning_rate         = best_params['learning_rate'],
        subsample             = best_params['subsample'],
        colsample_bytree      = best_params['colsample_bytree'],
        colsample_bylevel     = best_params['colsample_bylevel'],
        reg_alpha             = best_params['reg_alpha'],
        reg_lambda            = best_params['reg_lambda'],
        gamma                 = best_params['gamma'],
        min_split_loss        = best_params['min_split_loss'],
        scale_pos_weight      = scale_pos_weight,
        **XGB_FIXED_PARAMS,
    )
    final_model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        verbose  = 100,
    )
    print(f"  Best iteration : {final_model.best_iteration}")
    print(f"  Best AUC score : {final_model.best_score:.6f}")
    return final_model


def evaluate(model, X_val, y_val):
    """Evaluate model  returns metrics dict and prints report."""
    y_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)

    # F1-optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores   = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_thresh = float(thresholds[np.argmax(f1_scores)])

    # Business threshold  catch at least 60% of defaulters
    business_thresh = 0.50
    for thresh, rec in zip(thresholds, recalls[:-1]):
        if rec >= 0.60:
            business_thresh = float(thresh)
            break

    print(f"\n  Validation ROC-AUC      : {roc_auc:.6f}")
    print(f"  F1-optimal threshold    : {best_thresh:.4f}")
    print(f"  60% recall threshold    : {business_thresh:.4f}")

    print("\n-- Default threshold (0.50) --")
    print(classification_report(y_val, (y_proba >= 0.50).astype(int)))

    print(f"\n-- F1-optimal threshold ({best_thresh:.4f}) --")
    print(classification_report(y_val, (y_proba >= best_thresh).astype(int)))

    return {
        'roc_auc'          : roc_auc,
        'best_threshold'   : best_thresh,
        'business_threshold': business_thresh,
        'y_proba'          : y_proba,
    }


def save_model(model, filename=None):
    """Save model to models/ directory."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, filename or MODEL_FILE)
    joblib.dump(model, save_path)
    print(f"\n  Model saved -> {save_path}")
    return save_path


def load_model(filename=None):
    """Load model from models/ directory."""
    load_path = os.path.join(MODEL_DIR, filename or MODEL_FILE)
    model = joblib.load(load_path)
    print(f"  Model loaded <- {load_path}")
    return model


def get_feature_importance(model, X_train, top_n=25):
    """Return top N features by importance."""
    import pandas as pd
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    return feat_imp.nlargest(top_n)


def run_training_pipeline(X, y):
    """
    Full training pipeline  call this to run everything end to end.

    Returns: final_model, metrics, best_params
    """
    print("=" * 55)
    print("TRAINING PIPELINE")
    print("=" * 55)

    # Split
    X_train, X_val, y_train, y_val, scale_pos_weight = split_data(X, y)

    # Hyperparameter search
    best_params, study = run_optuna(X_train, y_train, scale_pos_weight)

    # Train final model
    final_model = train_final_model(
        X_train, y_train, X_val, y_val, best_params, scale_pos_weight
    )

    # Evaluate
    metrics = evaluate(final_model, X_val, y_val)

    # Feature importance
    print("\nTop 25 features:")
    print(get_feature_importance(final_model, X_train).to_string())

    # Save
    save_model(final_model)

    print("\n" + "=" * 55)
    print(f"DONE  ROC-AUC: {metrics['roc_auc']:.6f}")
    print("=" * 55)

    return final_model, metrics, best_params

def run_training_pipeline_with_mlflow(X, y, run_name="xgboost-optuna"):
    """
    Same as run_training_pipeline() but logs everything to MLflow.
    View results with: mlflow ui --backend-store-uri sqlite:///mlflow.db
    """
    #  SQLite backend  modern, no path issues, no deprecation warning
    mlflow.set_tracking_uri("sqlite:///C:/Project 2 - XGBoost/credit-risk-prediction/mlflow.db")
    mlflow.set_experiment("home-credit-default-risk")

    with mlflow.start_run(run_name=run_name):
        X_train, X_val, y_train, y_val, scale_pos_weight = split_data(X, y)
        mlflow.log_params({
            "train_size"       : len(X_train),
            "val_size"         : len(X_val),
            "n_features"       : X_train.shape[1],
            "scale_pos_weight" : round(scale_pos_weight, 2),
            "n_trials"         : N_TRIALS,
            "cv_folds"         : CV_FOLDS,
        })

        # -- Optuna search ----------------------------------
        best_params, study = run_optuna(X_train, y_train, scale_pos_weight)

        # Log best Optuna params
        mlflow.log_params(best_params)
        mlflow.log_metric("optuna_best_auc", study.best_value)

        # -- Train final model ------------------------------
        final_model = train_final_model(
            X_train, y_train, X_val, y_val, best_params, scale_pos_weight
        )

        # -- Evaluate ---------------------------------------
        metrics = evaluate(final_model, X_val, y_val)

        # Log all metrics
        mlflow.log_metrics({
            "val_roc_auc"        : round(metrics["roc_auc"], 6),
            "best_threshold"     : round(metrics["best_threshold"], 4),
            "business_threshold" : round(metrics["business_threshold"], 4),
            "best_iteration"     : final_model.best_iteration,
        })

        # -- Feature importance as artifact -----------------
        feat_imp = get_feature_importance(final_model, X_train)
        feat_imp_path = os.path.join(MODEL_DIR, "feature_importance.csv")
        feat_imp.to_csv(feat_imp_path)
        mlflow.log_artifact(feat_imp_path)

        # -- Log model --------------------------------------
        mlflow.xgboost.log_model(final_model, "model")

        # -- Save locally too -------------------------------
        save_model(final_model)

        print(f"\n  MLflow run logged successfully")
        print(f"  Run ID : {mlflow.active_run().info.run_id}")
        print(f"  View   : run 'mlflow ui' in terminal")

    return final_model, metrics, best_params
