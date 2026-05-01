"""
Master feature pipeline.

Usage:
    from src.features.pipeline import build_features

    X_train, X_test = build_features(train_df, test_df)

CV-safe target encoding is applied INSIDE cross-validation folds.
Never call fit_target_encoder on the full training set before splitting.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .race_features import RaceFeatures
from .tyre_features import TyreFeatures
from .driver_features import DriverFeatures

# Columns to drop before modelling
_DROP = ["id", "PitNextLap", "Race", "Driver", "Compound", "Year", "race_uid"]

# Categorical columns for label encoding
_CATS = ["Compound", "Race"]  # Driver handled separately (high cardinality)


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    drop_testing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.Series, list[str]]:
    """
    Apply full feature pipeline to train (and optionally test) data.

    Returns:
        X_train   : feature matrix
        X_test    : feature matrix (None if test_df not provided)
        y_train   : target series
        feature_names : list of column names
    """

    # ── 1. Clean + race features ──────────────────────────────────────────────
    race_transformer = RaceFeatures(drop_testing=drop_testing)
    train = race_transformer.transform(train_df)
    test  = race_transformer.transform(test_df) if test_df is not None else None

    # ── 2. Tyre features ──────────────────────────────────────────────────────
    tyre_transformer = TyreFeatures()
    train = tyre_transformer.transform(train)
    test  = tyre_transformer.transform(test) if test is not None else None

    # ── 3. Driver / position features ────────────────────────────────────────
    driver_transformer = DriverFeatures()
    train = driver_transformer.transform(train)
    test  = driver_transformer.transform(test) if test is not None else None

    # ── 4. Label encode low-cardinality categoricals ─────────────────────────
    for col in _CATS:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).unique()) + ["__unknown__"])
        train[f"{col}_enc"] = le.transform(train[col].astype(str))
        if test is not None:
            test_vals = test[col].astype(str).apply(
                lambda x: x if x in le.classes_ else "__unknown__"
            )
            test[f"{col}_enc"] = le.transform(test_vals)

    # ── 5. Driver: frequency encoding (safe for high-cardinality) ────────────
    driver_freq = train["Driver"].value_counts(normalize=True)
    train["driver_freq_enc"] = train["Driver"].map(driver_freq).fillna(0)
    if test is not None:
        test["driver_freq_enc"] = test["Driver"].map(driver_freq).fillna(0)

    # ── 6. Extract target ─────────────────────────────────────────────────────
    y_train = train["PitNextLap"].astype(int)

    # ── 7. Drop non-feature columns ───────────────────────────────────────────
    drop_cols = [c for c in _DROP if c in train.columns]
    X_train = train.drop(columns=drop_cols)
    X_test  = test.drop(columns=[c for c in _DROP if c in test.columns]) \
               if test is not None else None

    # Remove any remaining object columns
    obj_cols = X_train.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        print(f"[pipeline] Dropping remaining object cols: {obj_cols}")
        X_train = X_train.drop(columns=obj_cols)
        if X_test is not None:
            X_test = X_test.drop(columns=[c for c in obj_cols if c in X_test.columns])

    feature_names = X_train.columns.tolist()
    print(f"[pipeline] Final feature count: {len(feature_names)}")
    print(f"[pipeline] Train shape: {X_train.shape} | Target positive rate: {y_train.mean():.3f}")

    return X_train, X_test, y_train, feature_names


# ── CV-safe target encoder (use inside fold loop only) ───────────────────────
class CVTargetEncoder:
    """
    Fit on train fold, transform both train fold and val fold.
    NEVER fit on the full dataset — that's leakage.

    Usage inside CV:
        enc = CVTargetEncoder(cols=["Driver"])
        enc.fit(X_fold_train, y_fold_train)
        X_fold_train = enc.transform(X_fold_train)
        X_fold_val   = enc.transform(X_fold_val)
    """

    def __init__(self, cols: list[str], smoothing: float = 20.0):
        self.cols      = cols
        self.smoothing = smoothing
        self._maps: dict[str, pd.Series] = {}
        self._global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CVTargetEncoder":
        self._global_mean = y.mean()
        for col in self.cols:
            stats = pd.DataFrame({"y": y.values}, index=X.index)
            stats[col] = X[col].values
            agg = stats.groupby(col)["y"].agg(["mean", "count"])
            # Smoothed estimate = (n * cat_mean + k * global_mean) / (n + k)
            smooth = (agg["count"] * agg["mean"] + self.smoothing * self._global_mean) \
                     / (agg["count"] + self.smoothing)
            self._maps[col] = smooth
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            X[f"{col}_target_enc"] = (
                X[col].map(self._maps[col]).fillna(self._global_mean)
            )
        return X