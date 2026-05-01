"""
Race-level features + data cleaning.

Key decisions based on EDA:
  - Drop 'Pre-Season Testing' rows (not real races)
  - Create race_uid = Race + "_" + Year (CV group key)
  - Flag non-F1 series rows (Stint > 4 or Driver not matching D### pattern)
  - Encode compound as ordinal (hardness proxy)
"""
import re
import pandas as pd
import numpy as np
from .base import BaseTransformer


# Compound softness: higher = softer = wears faster
COMPOUND_SOFTNESS = {
    "SOFT": 4,
    "MEDIUM": 3,
    "HARD": 2,
    "INTERMEDIATE": 1,
    "WET": 0,
}

# Circuits known for high degradation (from F1 domain knowledge)
HIGH_DEG_CIRCUITS = {
    "Bahrain Grand Prix",
    "Spanish Grand Prix",
    "British Grand Prix",
    "Belgian Grand Prix",
    "Hungarian Grand Prix",
    "Abu Dhabi Grand Prix",
}


class RaceFeatures(BaseTransformer):
    """
    Adds race-level features and cleans data.
    Always call this first in the pipeline.
    """

    def __init__(self, drop_testing: bool = True):
        self.drop_testing = drop_testing

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Drop Pre-Season Testing (not a real race)
        if self.drop_testing:
            before = len(df)
            df = df[df["Race"] != "Pre-Season Testing"].reset_index(drop=True)
            print(f"[RaceFeatures] Dropped Pre-Season Testing: {before - len(df):,} rows removed")

        # 2. Unique race identifier for CV grouping
        df["race_uid"] = df["Race"].str.replace(" ", "_") + "_" + df["Year"].astype(str)

        # 3. Compound softness (ordinal)
        df["compound_softness"] = df["Compound"].map(COMPOUND_SOFTNESS).fillna(2)

        # 4. Is wet/intermediate flag (strategy wildcard)
        df["is_wet_compound"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)

        # 5. High degradation circuit flag
        df["high_deg_circuit"] = df["Race"].isin(HIGH_DEG_CIRCUITS).astype(int)

        # 6. Race progress buckets (strategy windows)
        #    0-25%: opening stint, 25-50%: first stop window,
        #    50-75%: second stop window, 75-100%: run to finish
        df["progress_bucket"] = pd.cut(
            df["RaceProgress"],
            bins=[0, 0.25, 0.50, 0.75, 1.01],
            labels=[0, 1, 2, 3],
            include_lowest=True,
        ).astype(int)

        # 7. Is final 10% of race (rarely pit here unless strategic)
        df["is_late_race"] = (df["RaceProgress"] >= 0.90).astype(int)

        # 8. Year encoded (data drift across seasons)
        df["year_encoded"] = df["Year"] - df["Year"].min()

        return df