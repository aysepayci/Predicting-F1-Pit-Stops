# F1 Pit Stop Prediction — Kaggle PS S6E5

**Competition:** Playground Series Season 6 Episode 5  
**Task:** Binary classification — predict `PitNextLap`  
**Metric:** ROC-AUC  

---

## Quick Start

```bash
pip install -r requirements.txt

# Run EDA
make eda

# Train baseline
make train

# Generate submission
make submit
```

---

## Leaderboard Progress

| Date | Model | CV AUC (mean ± std) | Public LB | Notes |
|------|-------|---------------------|-----------|-------|
| - | Baseline (LR) | - | - | initial pipeline |

---

## Key Findings

- **Class imbalance:** ~X% positive rate → use `scale_pos_weight`
- **CV strategy:** StratifiedGroupKFold, groups=`race_uid` (Race + Year)
- **Top features (SHAP):** TyreLife, RaceProgress, LapTime_Delta
- **Leakage risk:** `Cumulative_Degradation` — verify computation window

---

## Experiment Log

All experiments tracked in MLflow / W&B. Key runs documented here.

---

## Repository Structure

```
src/
  data/         — loading, validation, EDA
  features/     — modular feature transformers
  models/       — model classes (LGBM, XGB, CatBoost, MLP)
  evaluation/   — CV, metrics, SHAP
  utils/        — config, logging, seeds
notebooks/      — EDA only, numbered
experiments/    — YAML configs per run
submissions/    — CSV outputs + score log
```

| 2025-05-02 | LightGBM v1 (baseline) | 0.932 ± 0.013 | - | first real model |