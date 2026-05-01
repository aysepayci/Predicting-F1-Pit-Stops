"""
Driver and position features.

Note on 887 drivers: mixed motorsport series in dataset.
We avoid per-driver historical aggregates at training time
(would require a separate historical lookback table).
Instead we use position-based strategic proxies.
Target encoding for Driver is handled in pipeline.py with CV-safe encoding.
"""
import pandas as pd
import numpy as np
from .base import BaseTransformer


class DriverFeatures(BaseTransformer):
    """Position and driver context features."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Position buckets: fighting for points vs midfield vs backmarker
        df["position_bucket"] = pd.cut(
            df["Position"],
            bins=[0, 3, 10, 15, 100],
            labels=[0, 1, 2, 3],   # podium, points, midfield, back
            include_lowest=True,
        ).astype(int)

        # 2. Is fighting for podium (top 3) — most aggressive strategy
        df["is_podium_fight"] = (df["Position"] <= 3).astype(int)

        # 3. Is outside points (11+) — more likely to gamble on pit
        df["outside_points"] = (df["Position"] > 10).astype(int)

        # 4. Position change sign: gaining vs losing positions
        df["gaining_positions"] = (df["Position_Change"] > 0).astype(int)
        df["losing_positions"]  = (df["Position_Change"] < 0).astype(int)

        # 5. Absolute position change (magnitude of pace differential)
        df["abs_position_change"] = df["Position_Change"].abs()

        # 6. Position × race progress interaction
        #    Leader late in race rarely pits; backmarker might for fastest lap
        df["pos_x_progress"] = df["Position"] * df["RaceProgress"]

        # 7. PitStop count so far (how many stops already made this race)
        df["already_pitted"] = (df["PitStop"] > 0).astype(int)
        df["pit_count_sq"]   = df["PitStop"] ** 2

        return df