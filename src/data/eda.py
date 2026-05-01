"""
EDA module for F1 Pit Stop prediction.
Run: python src/data/eda.py --data_path data/raw/train.csv --output_dir reports/eda
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "xtick.color": "#999",
    "ytick.color": "#999",
    "text.color": "#eee",
    "grid.color": "#2a2a2a",
    "grid.linestyle": "--",
    "font.family": "monospace",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})
RED   = "#E24B4A"
AMBER = "#EF9F27"
TEAL  = "#1D9E75"
BLUE  = "#378ADD"
GRAY  = "#888780"


# ── Loader ────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


# ── 1. Dataset overview ───────────────────────────────────────────────────────
def overview(df: pd.DataFrame) -> dict:
    target    = "PitNextLap"
    pos_rate  = df[target].mean()
    null_pct  = (df.isnull().sum() / len(df) * 100).round(2)
    n_drivers = df["Driver"].nunique()
    n_races   = df["Race"].nunique()
    n_years   = df["Year"].nunique()

    stats = {
        "rows": len(df), "cols": len(df.columns),
        "pos_rate": pos_rate, "null_pct": null_pct,
        "n_drivers": n_drivers, "n_races": n_races, "n_years": n_years,
    }

    print("\n── Dataset Overview ──────────────────────────────")
    print(f"  Rows          : {stats['rows']:,}")
    print(f"  Columns       : {stats['cols']}")
    print(f"  Drivers       : {n_drivers}")
    print(f"  Races         : {n_races}")
    print(f"  Years         : {n_years}  ({df['Year'].min()}-{df['Year'].max()})")
    print(f"\n  Target (PitNextLap) positive rate: {pos_rate:.4f}  ({pos_rate*100:.2f}%)")
    print(f"  -> Class imbalance ratio  1 : {(1-pos_rate)/pos_rate:.1f}")

    nulls = null_pct[null_pct > 0]
    if len(nulls):
        print(f"\n  Null columns:\n{nulls.to_string()}")
    else:
        print("\n  No null values found.")

    return stats


# ── 2. Target distribution ────────────────────────────────────────────────────
def plot_target(df: pd.DataFrame, ax: plt.Axes) -> None:
    counts = df["PitNextLap"].value_counts().sort_index()
    bars = ax.bar(["No Pit (0)", "Pit Next Lap (1)"], counts.values,
                  color=[TEAL, RED], width=0.5, edgecolor="#111", linewidth=0.8)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
                f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom",
                fontsize=9, color="#ccc")
    ax.set_title("Target Distribution — PitNextLap")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(0, counts.max() * 1.2)


# ── 3. Pit rate by compound ───────────────────────────────────────────────────
def plot_compound_pitstop(df: pd.DataFrame, ax: plt.Axes) -> None:
    rates  = df.groupby("Compound")["PitNextLap"].mean().sort_values(ascending=False)
    avg    = df["PitNextLap"].mean()
    colors = [AMBER if r > avg else GRAY for r in rates.values]
    bars   = ax.barh(rates.index, rates.values, color=colors,
                     edgecolor="#111", linewidth=0.6)
    ax.axvline(avg, color=RED, linestyle="--", lw=1.2, label="Overall avg")
    ax.set_title("Pit Rate by Tyre Compound")
    ax.set_xlabel("P(PitNextLap)")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, rates.values):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)


# ── 4. TyreLife vs pit probability ───────────────────────────────────────────
def plot_tyrelife_vs_pit(df: pd.DataFrame, ax: plt.Axes) -> None:
    tl_bins    = pd.cut(df["TyreLife"], bins=30)
    rate       = df.groupby(tl_bins, observed=True)["PitNextLap"].mean()
    midpoints  = [iv.mid for iv in rate.index]
    ax.plot(midpoints, rate.values, color=AMBER, lw=1.8)
    ax.fill_between(midpoints, rate.values, alpha=0.2, color=AMBER)
    ax.set_title("Pit Probability vs Tyre Life")
    ax.set_xlabel("TyreLife (laps on compound)")
    ax.set_ylabel("P(PitNextLap)")
    ax.grid(alpha=0.3)


# ── 5. Pit rate by race progress ─────────────────────────────────────────────
def plot_race_progress(df: pd.DataFrame, ax: plt.Axes) -> None:
    rp_bins   = pd.cut(df["RaceProgress"], bins=20)
    rate      = df.groupby(rp_bins, observed=True)["PitNextLap"].mean()
    midpoints = [iv.mid for iv in rate.index]
    ax.plot(midpoints, rate.values, color=BLUE, lw=1.8)
    ax.fill_between(midpoints, rate.values, alpha=0.2, color=BLUE)
    for xv, label in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%")]:
        ax.axvline(xv, color=GRAY, linestyle=":", lw=1, label=label)
    ax.set_title("Pit Probability vs Race Progress")
    ax.set_xlabel("RaceProgress (0=start -> 1=finish)")
    ax.set_ylabel("P(PitNextLap)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


# ── 6. Correlation heatmap ────────────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame, ax: plt.Axes) -> None:
    num_cols = [
        "TyreLife", "LapNumber", "Stint", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change", "PitStop", "PitNextLap",
    ]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0, ax=ax,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3, linecolor="#111",
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Feature Correlation Matrix")


# ── 7. Top races by pit rate ──────────────────────────────────────────────────
def plot_race_pitstop_rate(df: pd.DataFrame, ax: plt.Axes) -> None:
    avg        = df["PitNextLap"].mean()
    race_stats = (
        df.groupby("Race")
        .agg(pit_rate=("PitNextLap", "mean"))
        .sort_values("pit_rate", ascending=True)
        .tail(15)
    )
    colors = [RED if r > avg * 1.2 else TEAL for r in race_stats["pit_rate"]]
    ax.barh(race_stats.index, race_stats["pit_rate"],
            color=colors, edgecolor="#111", linewidth=0.5)
    ax.axvline(avg, color=AMBER, linestyle="--", lw=1.2, label="Overall avg")
    ax.set_title("Top 15 Races by Pit Rate")
    ax.set_xlabel("P(PitNextLap)")
    ax.legend(fontsize=8)


# ── 8. Position vs pit ────────────────────────────────────────────────────────
def plot_position_vs_pit(df: pd.DataFrame, ax: plt.Axes) -> None:
    pos_rate = df[df["Position"] <= 20].groupby("Position")["PitNextLap"].mean()
    colors   = [RED if p <= 3 else BLUE if p <= 10 else GRAY for p in pos_rate.index]
    ax.bar(pos_rate.index, pos_rate.values,
           color=colors, edgecolor="#111", linewidth=0.5)
    ax.axhline(df["PitNextLap"].mean(), color=AMBER,
               linestyle="--", lw=1, label="Avg")
    ax.set_title("Pit Rate by Race Position")
    ax.set_xlabel("Position")
    ax.set_ylabel("P(PitNextLap)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


# ── Master runner ─────────────────────────────────────────────────────────────
def run_eda(data_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df    = load_data(data_path)
    stats = overview(df)

    fig = plt.figure(figsize=(20, 22), facecolor="#0f0f0f")
    fig.suptitle(
        "F1 Pit Stop Prediction — EDA Report  |  Kaggle Playground Series S6E5",
        fontsize=14, color="#eee", y=0.98, fontfamily="monospace",
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    plot_target(df,            fig.add_subplot(gs[0, 0]))
    plot_compound_pitstop(df,  fig.add_subplot(gs[0, 1]))
    plot_tyrelife_vs_pit(df,   fig.add_subplot(gs[1, 0]))
    plot_race_progress(df,     fig.add_subplot(gs[1, 1]))
    plot_correlation(df,       fig.add_subplot(gs[2, :]))
    plot_race_pitstop_rate(df, fig.add_subplot(gs[3, 0]))
    plot_position_vs_pit(df,   fig.add_subplot(gs[3, 1]))

    out_path = out / "eda_report.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"\n[saved] {out_path}")

    print("\n── Key Findings ──────────────────────────────────")
    print(f"  1. Class imbalance: {stats['pos_rate']*100:.1f}% positive")
    print(f"     -> use scale_pos_weight={((1-stats['pos_rate'])/stats['pos_rate']):.1f} in LightGBM")
    print(f"  2. {stats['n_drivers']} drivers | {stats['n_races']} races | {stats['n_years']} years")
    print(f"  3. TyreLife & RaceProgress likely strongest features (confirm with SHAP)")
    print(f"  4. CV strategy: StratifiedGroupKFold(groups=Race+Year, n_splits=5)")
    print(f"  5. Check Cumulative_Degradation — may encode future lap info (leakage risk)")
    print(f"  6. LapTime_Delta large negatives (e.g. -223s) — check for outliers / SC laps")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Pit Stop EDA")
    parser.add_argument("--data_path",  default="data/raw/train.csv")
    parser.add_argument("--output_dir", default="reports/eda")
    args = parser.parse_args()
    run_eda(args.data_path, args.output_dir)