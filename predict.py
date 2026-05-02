"""
Inference script — generates Kaggle submission.
Run: python predict.py
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

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/4] Building features...")

    # Kaggle expects ALL 188,165 rows.
    # Pre-Season Testing rows → prediction = 0.0 (no strategic pit stops in testing)
    is_testing = test_df["Race"] == "Pre-Season Testing"
    full_submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: 0.0})
    print(f"      Pre-Season Testing rows (pred=0.0): {is_testing.sum():,}")
    print(f"      Real race rows (will be predicted) : {(~is_testing).sum():,}")

    X_train, X_test, y_train, feature_names = build_features(
        train_df, test_df=test_df, drop_testing=True
    )
    print(f"      X_train: {X_train.shape} | X_test: {X_test.shape}")

    # ── 3. Full retrain ───────────────────────────────────────────────────────
    print("\n[3/4] Training on full dataset...")
    config = LGBMConfig(n_estimators=450, learning_rate=0.05, early_stopping_rounds=999)
    val_size = max(1000, int(len(X_train) * 0.05))
    model = LGBMModel(config)
    model.fit(
        X_train.iloc[:-val_size], y_train.iloc[:-val_size],
        X_train.iloc[-val_size:], y_train.iloc[-val_size:],
    )
    print(f"      Training complete. Best iter: {model.best_iteration_}")

    # ── 4. Predict ────────────────────────────────────────────────────────────
    print("\n[4/4] Generating predictions...")
    preds = model.predict_proba(X_test)
    print(f"      Pred range : [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"      Pred mean  : {preds.mean():.4f}  (train pos rate: {y_train.mean():.4f})")

    # Fill real-race predictions into the full template
    real_ids = test_df.loc[~is_testing, ID_COL].values
    full_submission.loc[full_submission[ID_COL].isin(real_ids), TARGET] = preds

    # ── 5. Save ───────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    sub_path  = SUBMISSIONS / f"sub_lgbm_v1_{timestamp}.csv"
    full_submission.to_csv(sub_path, index=False)

    print(f"\n[saved] {sub_path}")
    print(f"        Total rows : {len(full_submission):,}  (expected: {len(test_df):,})")
    print(f"        Head:\n{full_submission.head(3).to_string(index=False)}")
    print(f"\nDone in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()