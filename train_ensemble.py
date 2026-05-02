"""
Ensemble training — LightGBM + XGBoost OOF blend.
Run: python train_ensemble.py

Strategy:
  1. Run CV for both models → get OOF predictions
  2. Find optimal blend weight via Optuna on OOF
  3. Retrain both on full data
  4. Blend test predictions with optimal weight
  5. Save submission

Expected LB improvement: +0.003-0.008 over single model
"""
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, ".")

from src.features.pipeline import build_features
from src.models.lgbm_model import LGBMModel, LGBMConfig
from src.models.xgb_model import XGBModel, XGBConfig
from src.evaluation.cv import run_cv
from src.utils.config import TRAIN_FILE, TEST_FILE, SUBMISSIONS, DATA_PROC, RANDOM_SEED, N_FOLDS, ID_COL, TARGET


def main() -> None:
    t_start = time.time()
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & features ────────────────────────────────────────────────────
    print("[1/6] Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    is_testing = test_df["Race"] == "Pre-Season Testing"
    full_submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: 0.0})

    print("\n[2/6] Building features...")
    X_train, X_test, y_train, feats = build_features(
        train_df, test_df=test_df, drop_testing=True
    )
    print(f"      Shape: {X_train.shape}")

    cleaned_mask = train_df["Race"] != "Pre-Season Testing"
    groups = (
        train_df.loc[cleaned_mask, "Race"].str.replace(" ", "_")
        + "_" + train_df.loc[cleaned_mask, "Year"].astype(str)
    ).reset_index(drop=True)

    # ── 2. LGBM CV ────────────────────────────────────────────────────────────
    print("\n[3/6] LightGBM CV...")
    lgbm_config = LGBMConfig(
        n_estimators=2000, learning_rate=0.0217, num_leaves=156,
        max_depth=9, min_child_samples=132, feature_fraction=0.663,
        bagging_fraction=0.611, bagging_freq=5,
        reg_alpha=0.0885, reg_lambda=0.597, scale_pos_weight=4.66,
        early_stopping_rounds=100,
    )
    lgbm_results = run_cv(
        X_train, y_train, groups,
        build_model_fn=lambda: LGBMModel(lgbm_config),
        n_splits=N_FOLDS, seed=RANDOM_SEED,
    )
    oof_lgbm = lgbm_results["oof_preds"]
    print(f"      LGBM OOF AUC: {lgbm_results['oof_auc']:.5f}")
    np.save(DATA_PROC / "oof_lgbm.npy", oof_lgbm)

    # ── 3. XGBoost CV ─────────────────────────────────────────────────────────
    print("\n[4/6] XGBoost CV...")
    xgb_config  = XGBConfig(
        n_estimators=2000, learning_rate=0.02, max_depth=7,
        min_child_weight=50, subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=4.66, early_stopping_rounds=100,
    )
    xgb_results = run_cv(
        X_train, y_train, groups,
        build_model_fn=lambda: XGBModel(xgb_config),
        n_splits=N_FOLDS, seed=RANDOM_SEED,
    )
    oof_xgb = xgb_results["oof_preds"]
    print(f"      XGB  OOF AUC: {xgb_results['oof_auc']:.5f}")
    np.save(DATA_PROC / "oof_xgb.npy", oof_xgb)

    # ── 4. Optimal blend weight ───────────────────────────────────────────────
    print("\n[5/6] Finding optimal blend weight...")

    def blend_objective(trial):
        w = trial.suggest_float("w_lgbm", 0.0, 1.0)
        blended = w * oof_lgbm + (1 - w) * oof_xgb
        return roc_auc_score(y_train, blended)

    study = optuna.create_study(direction="maximize")
    study.optimize(blend_objective, n_trials=200, show_progress_bar=False)
    w_lgbm = study.best_params["w_lgbm"]
    w_xgb  = 1 - w_lgbm

    oof_blend = w_lgbm * oof_lgbm + w_xgb * oof_xgb
    blend_auc = roc_auc_score(y_train, oof_blend)

    print(f"      Blend weights: LGBM={w_lgbm:.3f}, XGB={w_xgb:.3f}")
    print(f"      LGBM  OOF AUC : {lgbm_results['oof_auc']:.5f}")
    print(f"      XGB   OOF AUC : {xgb_results['oof_auc']:.5f}")
    print(f"      BLEND OOF AUC : {blend_auc:.5f}  <- submit this")

    # ── 5. Full retrain + predict ─────────────────────────────────────────────
    print("\n[6/6] Full retrain and predict...")
    val_size = max(1000, int(len(X_train) * 0.05))

    # LGBM
    lgbm_full = LGBMModel(LGBMConfig(
        n_estimators=int(np.mean([m.best_iteration_ for m in lgbm_results["models"]])),
        learning_rate=0.0217, num_leaves=156, max_depth=9,
        min_child_samples=132, feature_fraction=0.663,
        bagging_fraction=0.611, bagging_freq=5,
        reg_alpha=0.0885, reg_lambda=0.597, scale_pos_weight=4.66,
        early_stopping_rounds=999,
    ))
    lgbm_full.fit(X_train.iloc[:-val_size], y_train.iloc[:-val_size],
                  X_train.iloc[-val_size:], y_train.iloc[-val_size:])
    preds_lgbm = lgbm_full.predict_proba(X_test)

    # XGB
    xgb_full = XGBModel(XGBConfig(
        n_estimators=int(np.mean([m.best_iteration_ for m in xgb_results["models"]])),
        learning_rate=0.02, max_depth=7, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.7, scale_pos_weight=4.66,
        early_stopping_rounds=999,
    ))
    xgb_full.fit(X_train.iloc[:-val_size], y_train.iloc[:-val_size],
                 X_train.iloc[-val_size:], y_train.iloc[-val_size:])
    preds_xgb = xgb_full.predict_proba(X_test)

    # Blend
    preds_blend = w_lgbm * preds_lgbm + w_xgb * preds_xgb

    # ── Save ──────────────────────────────────────────────────────────────────
    real_ids = test_df.loc[~is_testing, ID_COL].values
    full_submission.loc[full_submission[ID_COL].isin(real_ids), TARGET] = preds_blend

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    sub_path  = SUBMISSIONS / f"sub_ensemble_lgbm_xgb_{timestamp}.csv"
    full_submission.to_csv(sub_path, index=False)

    # Save blend config
    blend_cfg = {
        "w_lgbm": w_lgbm, "w_xgb": w_xgb,
        "oof_lgbm_auc": lgbm_results["oof_auc"],
        "oof_xgb_auc":  xgb_results["oof_auc"],
        "oof_blend_auc": blend_auc,
    }
    with open("experiments/ensemble_blend_weights.json", "w") as f:
        json.dump(blend_cfg, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Blend OOF AUC : {blend_auc:.5f}")
    print(f"  Submission    : {sub_path.name}")
    print(f"  Total time    : {time.time()-t_start:.1f}s")
    print(f"{'='*50}")
    print("\nNext: submit to Kaggle and record LB score.")


if __name__ == "__main__":
    main()