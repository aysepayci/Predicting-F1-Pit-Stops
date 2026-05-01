"""
Cross-validation loop — the most important file in the project.

Strategy: StratifiedGroupKFold
  - Stratified  : maintains target ratio across folds (~20% positive)
  - Group       : all laps from the same race stay in one fold
                  → prevents race-level data leakage

Never use random KFold on this data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from typing import Callable, Any
import time


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    build_model_fn: Callable[[], Any],
    n_splits: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run StratifiedGroupKFold CV.

    Args:
        X              : feature matrix (from build_features)
        y              : target series (PitNextLap)
        groups         : race_uid series — one group per race
        build_model_fn : callable that returns a fresh unfitted model
                         e.g. lambda: LGBMModel(LGBMConfig())
        n_splits       : number of CV folds (default 5)
        seed           : random seed
        verbose        : print fold scores

    Returns:
        dict with keys:
            oof_preds  : np.array of out-of-fold predictions (same length as X)
            fold_aucs  : list of per-fold AUC scores
            mean_auc   : float
            std_auc    : float
            models     : list of fitted model objects (for ensemble)
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds  = np.zeros(len(X))
    fold_aucs  = []
    models     = []
    fold_times = []

    if verbose:
        print(f"\n{'='*55}")
        print(f"  StratifiedGroupKFold CV  |  {n_splits} folds")
        print(f"  Groups (unique races): {groups.nunique()}")
        print(f"  Target positive rate : {y.mean():.4f}")
        print(f"{'='*55}")

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(X, y, groups), start=1
    ):
        t0 = time.time()

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Sanity check: no group leakage
        tr_groups  = set(groups.iloc[train_idx].unique())
        val_groups = set(groups.iloc[val_idx].unique())
        assert tr_groups.isdisjoint(val_groups), \
            f"Fold {fold}: train/val group overlap detected — leakage!"

        # Train
        model = build_model_fn()
        model.fit(X_tr, y_tr, X_val, y_val)

        # Predict
        val_preds             = model.predict_proba(X_val)
        oof_preds[val_idx]    = val_preds
        fold_auc              = roc_auc_score(y_val, val_preds)
        fold_aucs.append(fold_auc)
        models.append(model)

        elapsed = time.time() - t0
        fold_times.append(elapsed)

        if verbose:
            best_iter = getattr(model, "best_iteration_", "n/a")
            print(
                f"  Fold {fold}/{n_splits}  "
                f"AUC={fold_auc:.5f}  "
                f"best_iter={best_iter}  "
                f"time={elapsed:.1f}s"
            )

    oof_auc  = roc_auc_score(y, oof_preds)
    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))

    if verbose:
        print(f"{'='*55}")
        print(f"  OOF  AUC : {oof_auc:.5f}")
        print(f"  Mean AUC : {mean_auc:.5f} ± {std_auc:.5f}")
        print(f"  Fold AUCs: {[round(a,5) for a in fold_aucs]}")
        print(f"  Total    : {sum(fold_times):.1f}s")
        print(f"{'='*55}\n")

    return {
        "oof_preds":  oof_preds,
        "oof_auc":    oof_auc,
        "fold_aucs":  fold_aucs,
        "mean_auc":   mean_auc,
        "std_auc":    std_auc,
        "models":     models,
    }