"""
Cross-validation loop.

Default: StratifiedKFold (correct for this competition — row-level random split)
Optional: StratifiedGroupKFold (kept for reference)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from typing import Callable, Any, Optional
import time


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    build_model_fn: Callable[[], Any],
    groups: Optional[pd.Series] = None,
    n_splits: int = 5,
    seed: int = 42,
    use_group_kfold: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run cross-validation.

    Args:
        X              : feature matrix
        y              : target
        build_model_fn : lambda returning a fresh unfitted model
        groups         : race_uid (only needed if use_group_kfold=True)
        n_splits       : folds
        seed           : random seed
        use_group_kfold: if True use StratifiedGroupKFold (old strategy)
                         if False use StratifiedKFold (correct for this comp)
    """
    if use_group_kfold:
        assert groups is not None
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_args = (X, y, groups)
        cv_type = "StratifiedGroupKFold"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_args = (X, y)
        cv_type = "StratifiedKFold"

    oof_preds  = np.zeros(len(X))
    fold_aucs  = []
    models     = []

    if verbose:
        print(f"\n{'='*55}")
        print(f"  {cv_type}  |  {n_splits} folds")
        print(f"  Target positive rate : {y.mean():.4f}")
        print(f"{'='*55}")

    for fold, (train_idx, val_idx) in enumerate(splitter.split(*split_args), 1):
        t0 = time.time()
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model_fn()
        model.fit(X_tr, y_tr, X_val, y_val)

        val_preds          = model.predict_proba(X_val)
        oof_preds[val_idx] = val_preds
        fold_auc           = roc_auc_score(y_val, val_preds)
        fold_aucs.append(fold_auc)
        models.append(model)

        if verbose:
            best_iter = getattr(model, "best_iteration_", "n/a")
            print(f"  Fold {fold}/{n_splits}  AUC={fold_auc:.5f}  "
                  f"best_iter={best_iter}  time={time.time()-t0:.1f}s")

    oof_auc  = roc_auc_score(y, oof_preds)
    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))

    if verbose:
        print(f"{'='*55}")
        print(f"  OOF  AUC : {oof_auc:.5f}")
        print(f"  Mean AUC : {mean_auc:.5f} ± {std_auc:.5f}")
        print(f"  Fold AUCs: {[round(a,5) for a in fold_aucs]}")
        print(f"{'='*55}\n")

    return {
        "oof_preds": oof_preds, "oof_auc": oof_auc,
        "fold_aucs": fold_aucs, "mean_auc": mean_auc,
        "std_auc":  std_auc,   "models":  models,
    }