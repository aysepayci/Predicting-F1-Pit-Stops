"""Base transformer — all feature classes inherit from this."""
from abc import ABC, abstractmethod
import pandas as pd


class BaseTransformer(ABC):
    """Sklearn-compatible transformer interface."""

    def fit(self, df: pd.DataFrame, y=None) -> "BaseTransformer":
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)