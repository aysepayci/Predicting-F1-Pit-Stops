"""
Optuna hyperparameter search for LightGBM.
Run: python tune.py

Strategy:
  - 50 trials (fast enough on 416K rows)
  - Same StratifiedGroupKFold as train.py (3-fold for speed)
  - Logs best params to experiments/lgbm_best_params.json
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, ".")

from src.features.pipeline import build_features
from src.utils.config import TRAIN_FILE, RANDOM_SEED


# ── Load & build features once (reused across all trials) ────────────────────
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
X, _, y, feats = build_features(train_df, drop_testing=True)

# Drop year_encoded — overfits to year distribution
if "year_encoded" in X.columns:
    X = X.drop(columns=["year_encoded"])
    print("Dropped year_encoded")

cleaned_mask = train_df["Race"] != "Pre-Season Testing"
groups = (
    train_df.loc[cleaned_mask, "Race"].str.replace(" ", "_")
    + "_"
    + train_df.loc[cleaned_mask, "Year"].astype(str)
).reset_index(drop=True)

print(f"Features: {X.shape[1]} | Rows: {X.shape[0]:,} | Groups: {groups.nunique()}")

# Use 3-fold for speed during tuning
TUNE_FOLDS  = 3
TUNE_SEED   = RANDOM_SEED
N_TRIALS    = 50


# ── Objective ─────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "objective":          "binary",
        "metric":             "auc",
        "boosting_type":      "gbdt",
        "verbosity":          -1,
        "n_jobs":             -1,
        "random_state":       TUNE_SEED,
        # ── Tuned params ──
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":         trial.suggest_int("num_leaves", 31, 255),
        "max_depth":          trial.suggest_int("max_depth", 4, 10),
        "min_child_samples":  trial.suggest_int("min_child_samples", 20, 200),
        "feature_fraction":   trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":   trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":       trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight":   trial.suggest_float("scale_pos_weight", 2.0, 6.0),
    }

    sgkf   = StratifiedGroupKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=TUNE_SEED)
    aucs   = []

    for train_idx, val_idx in sgkf.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        aucs.append(roc_auc_score(y_val, preds))

    return float(np.mean(aucs))


# ── Run study ─────────────────────────────────────────────────────────────────
print(f"\nStarting Optuna search — {N_TRIALS} trials, {TUNE_FOLDS}-fold CV")
print("This will take ~15-25 minutes...\n")

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=TUNE_SEED),
    study_name="lgbm_f1_pitstop",
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ── Results ───────────────────────────────────────────────────────────────────
best = study.best_trial
print(f"\n{'='*50}")
print(f"Best CV AUC : {best.value:.5f}")
print(f"Best params :")
for k, v in best.params.items():
    print(f"  {k:<25} = {v}")
print(f"{'='*50}")

# Save
Path("experiments").mkdir(exist_ok=True)
result = {"best_auc": best.value, "params": best.params}
out_path = Path("experiments/lgbm_best_params.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {out_path}")
print("Next: update LGBMConfig in train.py with these params, then retrain.")