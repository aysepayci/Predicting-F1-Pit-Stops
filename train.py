"""
Main training script.
Run: python train.py

Outputs:
  - OOF predictions : data/processed/oof_lgbm_v1.npy
  - Feature importance plot saved to reports/
  - CV results printed to console
"""
import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.features.pipeline import build_features
from src.models.lgbm_model import LGBMModel, LGBMConfig
from src.evaluation.cv import run_cv
from src.utils.config import (
    TRAIN_FILE, DATA_PROC, RANDOM_SEED, N_FOLDS, TARGET, GROUP_COL
)


def main() -> None:
    t_start = time.time()
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("[1/4] Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    print(f"      Raw shape: {train_df.shape}")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/4] Building features...")
    X_train, _, y_train, feature_names = build_features(
        train_df, test_df=None, drop_testing=True
    )

    # Groups for CV — must be built AFTER drop_testing cleans the data
    # Re-create race_uid from the cleaned train_df
    cleaned_mask = train_df["Race"] != "Pre-Season Testing"
    groups = (
        train_df.loc[cleaned_mask, "Race"].str.replace(" ", "_")
        + "_"
        + train_df.loc[cleaned_mask, "Year"].astype(str)
    ).reset_index(drop=True)

    print(f"      Features  : {len(feature_names)}")
    print(f"      Groups    : {groups.nunique()} unique races")

    # ── 3. CV ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Running cross-validation...")

    results = run_cv(
        X         = X_train,
        y         = y_train,
        groups    = groups,
        build_model_fn = lambda: LGBMModel(LGBMConfig()),
        n_splits  = N_FOLDS,
        seed      = RANDOM_SEED,
        verbose   = True,
    )

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    print("[4/4] Saving outputs...")

    # OOF predictions
    oof_path = DATA_PROC / "oof_lgbm_v1.npy"
    np.save(oof_path, results["oof_preds"])
    print(f"      OOF saved  : {oof_path}")

    # Feature importance (average over folds)
    importance_frames = []
    for model in results["models"]:
        importance_frames.append(model.get_feature_importance(top_n=len(feature_names)))
    mean_importance = pd.concat(importance_frames, axis=1).mean(axis=1).sort_values(ascending=False)

    imp_path = Path("reports") / "feature_importance_lgbm_v1.csv"
    mean_importance.to_csv(imp_path, header=["importance"])
    print(f"      Importance : {imp_path}")

    # Results summary
    summary = {
        "model":      "lgbm_v1",
        "oof_auc":    round(results["oof_auc"], 6),
        "mean_auc":   round(results["mean_auc"], 6),
        "std_auc":    round(results["std_auc"], 6),
        "fold_aucs":  [round(a, 6) for a in results["fold_aucs"]],
        "n_features": len(feature_names),
        "elapsed_s":  round(time.time() - t_start, 1),
    }
    summary_path = Path("reports") / "cv_results_lgbm_v1.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      Summary    : {summary_path}")

    # Print top 15 features
    print("\n── Top 15 Features (gain importance) ──────────────")
    for feat, score in mean_importance.head(15).items():
        bar = "█" * int(score / mean_importance.max() * 30)
        print(f"  {feat:<35} {bar}")

    print(f"\nDone in {time.time() - t_start:.1f}s")
    print(f"OOF AUC: {results['oof_auc']:.5f}")


if __name__ == "__main__":
    main()