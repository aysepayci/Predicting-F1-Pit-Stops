"""
Inference script — generates Kaggle submission.
Run: python predict.py

Workflow:
  1. Load train + test
  2. Build features (same pipeline as train.py)
  3. Retrain on FULL training data (no CV — use all signal)
  4. Predict on test
  5. Save submissions/sub_YYYYMMDD_HHMMSS.csv
"""
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.features.pipeline import build_features
from src.models.lgbm_model import LGBMModel, LGBMConfig
from src.utils.config import TRAIN_FILE, TEST_FILE, SUBMISSIONS, TARGET, ID_COL


def main() -> None:
    t_start = time.time()
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("[1/4] Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    print(f"      Train: {train_df.shape} | Test: {test_df.shape}")

    # Keep test IDs before pipeline (pipeline may drop/reindex)
    test_ids = test_df[ID_COL].copy()

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/4] Building features...")
    X_train, X_test, y_train, feature_names = build_features(
        train_df, test_df=test_df, drop_testing=True
    )
    print(f"      X_train: {X_train.shape} | X_test: {X_test.shape}")

    # Align test IDs after Pre-Season Testing rows are dropped from train
    # Test set keeps all its rows (no Pre-Season Testing in test typically)
    assert X_test is not None, "X_test should not be None"
    assert len(X_test) == len(test_ids), \
        f"Test size mismatch: {len(X_test)} features vs {len(test_ids)} ids"

    # ── 3. Full retrain ───────────────────────────────────────────────────────
    print("\n[3/4] Training on full dataset...")
    print("      (No validation set — using best_iter from CV: ~450 avg)")

    # Use average best_iteration from CV runs as fixed n_estimators
    # This avoids needing a val set while using the CV-tuned stopping point
    config = LGBMConfig(
        n_estimators=450,        # avg best_iter from 5-fold CV
        learning_rate=0.05,
        early_stopping_rounds=999,  # effectively disabled
    )

    # Trick: use 5% of train as a dummy val to satisfy lgb.train API
    # but ignore early stopping (n_estimators is already fixed)
    val_size  = max(1000, int(len(X_train) * 0.05))
    X_val_dummy = X_train.iloc[-val_size:]
    y_val_dummy = y_train.iloc[-val_size:]
    X_tr_main   = X_train.iloc[:-val_size]
    y_tr_main   = y_train.iloc[:-val_size]

    model = LGBMModel(config)
    model.fit(X_tr_main, y_tr_main, X_val_dummy, y_val_dummy)
    print(f"      Training complete. Best iter: {model.best_iteration_}")

    # ── 4. Predict ────────────────────────────────────────────────────────────
    print("\n[4/4] Generating predictions...")
    preds = model.predict_proba(X_test)
    print(f"      Pred range : [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"      Pred mean  : {preds.mean():.4f}  (train positive rate: {y_train.mean():.4f})")

    # ── 5. Save submission ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    sub_path  = SUBMISSIONS / f"sub_lgbm_v1_{timestamp}.csv"

    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET: preds,
    })
    submission.to_csv(sub_path, index=False)

    print(f"\n[saved] {sub_path}")
    print(f"        Rows: {len(submission):,}")
    print(f"        Head:\n{submission.head(3).to_string(index=False)}")
    print(f"\nDone in {time.time() - t_start:.1f}s")
    print(f"\nNext step: submit {sub_path.name} to Kaggle and record public LB score.")


if __name__ == "__main__":
    main()