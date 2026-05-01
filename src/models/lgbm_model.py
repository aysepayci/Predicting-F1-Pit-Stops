"""
LightGBM model wrapper.
Inherits BaseModel interface — same fit/predict/evaluate for all models.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LGBMConfig:
    """All hyperparameters in one place. Override in experiments/lgbm_v1.yaml"""
    objective:        str   = "binary"
    metric:           str   = "auc"
    boosting_type:    str   = "gbdt"
    n_estimators:     int   = 1000
    learning_rate:    float = 0.05
    num_leaves:       int   = 63
    max_depth:        int   = -1
    min_child_samples:int   = 50
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq:     int   = 1
    reg_alpha:        float = 0.1
    reg_lambda:       float = 1.0
    scale_pos_weight: float = 4.0   # ~1:4 imbalance from EDA
    random_state:     int   = 42
    n_jobs:           int   = -1
    verbose:          int   = -1
    early_stopping_rounds: int = 50


class LGBMModel:
    """
    LightGBM binary classifier with early stopping.

    Usage:
        model = LGBMModel(LGBMConfig())
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict_proba(X_test)
        print(model.best_iteration_)
    """

    def __init__(self, config: Optional[LGBMConfig] = None):
        self.config = config or LGBMConfig()
        self.model_: Optional[lgb.Booster] = None
        self.best_iteration_: int = 0
        self.feature_importances_: Optional[pd.Series] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "LGBMModel":
        cfg = self.config

        params = {
            "objective":         cfg.objective,
            "metric":            cfg.metric,
            "boosting_type":     cfg.boosting_type,
            "learning_rate":     cfg.learning_rate,
            "num_leaves":        cfg.num_leaves,
            "max_depth":         cfg.max_depth,
            "min_child_samples": cfg.min_child_samples,
            "feature_fraction":  cfg.feature_fraction,
            "bagging_fraction":  cfg.bagging_fraction,
            "bagging_freq":      cfg.bagging_freq,
            "reg_alpha":         cfg.reg_alpha,
            "reg_lambda":        cfg.reg_lambda,
            "scale_pos_weight":  cfg.scale_pos_weight,
            "random_state":      cfg.random_state,
            "n_jobs":            cfg.n_jobs,
            "verbose":           cfg.verbose,
        }

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        self.model_ = lgb.train(
            params,
            dtrain,
            num_boost_round=cfg.n_estimators,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        self.best_iteration_ = self.model_.best_iteration
        self.feature_importances_ = pd.Series(
            self.model_.feature_importance(importance_type="gain"),
            index=X_train.columns,
        ).sort_values(ascending=False)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None, "Call fit() first"
        return self.model_.predict(X, num_iteration=self.best_iteration_)

    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        assert self.feature_importances_ is not None
        return self.feature_importances_.head(top_n)