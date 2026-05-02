"""
XGBoost model wrapper — same interface as LGBMModel.
Diversity source for ensemble: XGB and LGBM make different errors.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from dataclasses import dataclass
from typing import Optional


@dataclass
class XGBConfig:
    n_estimators:     int   = 2000
    learning_rate:    float = 0.02
    max_depth:        int   = 7
    min_child_weight: int   = 50
    subsample:        float = 0.8
    colsample_bytree: float = 0.7
    reg_alpha:        float = 0.1
    reg_lambda:       float = 1.0
    scale_pos_weight: float = 4.66
    random_state:     int   = 42
    n_jobs:           int   = -1
    early_stopping_rounds: int = 100
    eval_metric:      str   = "auc"
    tree_method:      str   = "hist"   # fast on CPU


class XGBModel:
    def __init__(self, config: Optional[XGBConfig] = None):
        self.config  = config or XGBConfig()
        self.model_: Optional[xgb.Booster] = None
        self.best_iteration_: int = 0
        self.feature_importances_: Optional[pd.Series] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
    ) -> "XGBModel":
        cfg = self.config
        params = {
            "objective":        "binary:logistic",
            "eval_metric":      cfg.eval_metric,
            "learning_rate":    cfg.learning_rate,
            "max_depth":        cfg.max_depth,
            "min_child_weight": cfg.min_child_weight,
            "subsample":        cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_alpha":        cfg.reg_alpha,
            "reg_lambda":       cfg.reg_lambda,
            "scale_pos_weight": cfg.scale_pos_weight,
            "seed":             cfg.random_state,
            "nthread":          cfg.n_jobs,
            "tree_method":      cfg.tree_method,
            "verbosity":        0,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
        dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=list(X_train.columns))

        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=cfg.n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=cfg.early_stopping_rounds,
            verbose_eval=100,
        )

        self.best_iteration_ = self.model_.best_iteration
        scores = self.model_.get_score(importance_type="gain")
        self.feature_importances_ = pd.Series(scores).sort_values(ascending=False)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None, "Call fit() first"
        dmat = xgb.DMatrix(X, feature_names=list(X.columns))
        return self.model_.predict(dmat, iteration_range=(0, self.best_iteration_))

    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        assert self.feature_importances_ is not None
        return self.feature_importances_.head(top_n)