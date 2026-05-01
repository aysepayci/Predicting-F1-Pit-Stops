"""
Tyre and stint features — highest expected predictive power.

Key insight: A pit stop is most likely when:
  1. TyreLife exceeds compound's expected lifetime
  2. Lap time degradation accelerates
  3. Stint is unusually long vs historical average

Leakage note: ALL features here use only current lap info (no future laps).
"""
import pandas as pd
import numpy as np
from .base import BaseTransformer


# Typical maximum stint lengths by compound (F1 domain knowledge)
# Used to compute "tyre life ratio" = TyreLife / expected_max
EXPECTED_MAX_STINT = {
    "SOFT": 20,
    "MEDIUM": 35,
    "HARD": 50,
    "INTERMEDIATE": 30,
    "WET": 40,
}


class TyreFeatures(BaseTransformer):
    """Tyre age, degradation, and stint-based features."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Tyre life ratio: how far through expected compound life
        expected = df["Compound"].map(EXPECTED_MAX_STINT).fillna(35)
        df["tyre_life_ratio"] = (df["TyreLife"] / expected).clip(0, 2)

        # 2. Tyre "overlife" flag: past expected maximum
        df["tyre_overlife"] = (df["TyreLife"] > expected).astype(int)

        # 3. Tyre life squared (nonlinear degradation curve)
        df["tyre_life_sq"] = df["TyreLife"] ** 2

        # 4. Tyre life × compound softness interaction
        softness = df.get("compound_softness", df["Compound"].map(
            {"SOFT": 4, "MEDIUM": 3, "HARD": 2, "INTERMEDIATE": 1, "WET": 0}
        ).fillna(2))
        df["tyre_life_x_softness"] = df["TyreLife"] * softness

        # 5. Stint number (later stints = more strategic flexibility)
        df["stint_is_final"] = (df["Stint"] >= 3).astype(int)

        # 6. Cumulative degradation per lap (average degradation rate this stint)
        # Guard against division by zero
        df["deg_per_lap"] = df["Cumulative_Degradation"] / (df["TyreLife"].clip(lower=1))

        # 7. Lap time delta clip: remove safety car / red flag outliers
        # Values beyond ±30s are likely SC laps, not real degradation
        df["lap_delta_clean"] = df["LapTime_Delta"].clip(-30, 30)

        # 8. Positive degradation flag (lap time increasing = tyre wearing)
        df["is_degrading"] = (df["LapTime_Delta"] > 0).astype(int)

        # 9. Acceleration of degradation proxy
        # LapTime_Delta > 1.5s per lap = significant drop-off
        df["strong_degradation"] = (df["lap_delta_clean"] > 1.5).astype(int)

        return df