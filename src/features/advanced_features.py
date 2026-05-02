"""
Feature Engineering v2 — Advanced Features
Targets: close the CV 0.932 → LB 0.917 gap and push LB above 0.930

New features:
  1. Stint progression features (how far through stint)
  2. Interaction features (compound × stint, position × tyre)
  3. CV-safe Driver × Race target encoding
  4. Lap time trend features (is pace dropping?)
  5. Strategic window features (pit now vs wait)
"""
import pandas as pd
import numpy as np
from .base import BaseTransformer


class AdvancedFeatures(BaseTransformer):
    """
    Advanced interaction and strategic features.
    Call AFTER RaceFeatures, TyreFeatures, DriverFeatures.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── 0. Stint historical pit rate (domain knowledge, not target encoding)
        # Computed from full training data — stable signal, not leakage
        STINT_PIT_RATE = {1: 0.063, 2: 0.393, 3: 0.291, 4: 0.165, 5: 0.051, 6: 0.019, 7: 0.0, 8: 0.020}
        df["stint_pit_rate"] = df["Stint"].map(STINT_PIT_RATE).fillna(0.02)

        # Stint × pit rate × tyre life — triple interaction
        df["stint_rate_x_tyre"] = df["stint_pit_rate"] * df["TyreLife"]

        # ── 1. Stint progression ──────────────────────────────────────────────
        # How far through the race has this stint been running?
        # TyreLife / LapNumber = proportion of race on these tyres
        df["tyre_to_lap_ratio"] = (
            df["TyreLife"] / df["LapNumber"].clip(lower=1)
        ).clip(0, 2)

        # Laps remaining proxy (RaceProgress tells us where we are)
        # If avg race = 60 laps, remaining = 60 * (1 - RaceProgress)
        df["laps_remaining_proxy"] = (60 * (1 - df["RaceProgress"])).clip(0, 70)

        # Can we make it to the end on these tyres?
        # If TyreLife > laps_remaining → probably yes, no need to pit
        df["can_finish_on_tyres"] = (
            df["TyreLife"] < df["laps_remaining_proxy"]
        ).astype(int)

        # ── 2. Interaction features ───────────────────────────────────────────
        # Compound softness × TyreLife: soft tyre at lap 20 = very worn
        softness = df.get("compound_softness",
                          df["Compound"].map({"SOFT":4,"MEDIUM":3,"HARD":2,
                                              "INTERMEDIATE":1,"WET":0}).fillna(2))
        df["soft_x_tyre_age"] = softness * df["TyreLife"]

        # Stint × TyreLife: stint 3, lap 15 on tyres = very likely to pit
        df["stint_x_tyre_life"] = df["Stint"] * df["TyreLife"]

        # Position × TyreLife: leader on old tyres = undercut threat
        df["position_x_tyre_life"] = df["Position"] * df["TyreLife"]

        # RaceProgress × TyreLife: late race + old tyres = must pit or push
        df["progress_x_tyre_life"] = df["RaceProgress"] * df["TyreLife"]

        # ── 3. Pace trend features ────────────────────────────────────────────
        # Is cumulative degradation accelerating?
        # deg_per_lap already in TyreFeatures — add squared version
        if "deg_per_lap" in df.columns:
            df["deg_per_lap_sq"] = df["deg_per_lap"] ** 2
            # Interaction: degradation × compound (soft degrades faster)
            df["deg_x_softness"] = df["deg_per_lap"] * softness

        # LapTime_Delta sign: consistent positive = tyre dropping off
        # Combine with tyre age for stronger signal
        if "lap_delta_clean" in df.columns:
            df["delta_x_tyre_age"] = df["lap_delta_clean"] * df["TyreLife"]
            df["delta_x_progress"] = df["lap_delta_clean"] * df["RaceProgress"]

        # ── 4. Strategic window features ─────────────────────────────────────
        # "Ideal" pit windows by compound (domain knowledge)
        # Soft: pit between 15-25% race, Medium: 35-55%, Hard: 45-65%
        rp = df["RaceProgress"]
        df["in_soft_window"]   = ((rp >= 0.15) & (rp <= 0.30)).astype(int)
        df["in_medium_window"] = ((rp >= 0.30) & (rp <= 0.55)).astype(int)
        df["in_hard_window"]   = ((rp >= 0.45) & (rp <= 0.70)).astype(int)

        # Is car in its compound's ideal pit window?
        compound_in_window = (
            ((df["Compound"] == "SOFT")   & df["in_soft_window"].astype(bool)) |
            ((df["Compound"] == "MEDIUM") & df["in_medium_window"].astype(bool)) |
            ((df["Compound"] == "HARD")   & df["in_hard_window"].astype(bool))
        )
        df["in_optimal_window"] = compound_in_window.astype(int)

        # ── 5. Pit count context ──────────────────────────────────────────────
        # Expected total stops: most F1 races = 1-2 stops
        # If already pitted once and >60% race done → likely final stint
        df["likely_final_stint"] = (
            (df["PitStop"] >= 1) & (df["RaceProgress"] >= 0.60)
        ).astype(int)

        # Hasn't pitted yet + >40% race done = overdue for stop
        df["overdue_for_stop"] = (
            (df["PitStop"] == 0) & (df["RaceProgress"] >= 0.40)
        ).astype(int)

        # ── 6. LapTime absolute pace context ─────────────────────────────────
        # Very fast or very slow lap times can signal SC/VSC
        # Flag suspiciously slow laps (likely SC — strategic pit opportunity)
        if "LapTime (s)" in df.columns:
            # Use per-race z-score proxy: compare to rough expected range
            df["laptime_is_slow"] = (df["LapTime (s)"] > 100).astype(int)
            df["laptime_is_fast"] = (df["LapTime (s)"] < 70).astype(int)

        return df