# UFC Fight Matchmaker

An ML-powered system that predicts fight entertainment value and suggests the
most exciting matchups across UFC weight classes.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the matchmaker on a division
python main.py matchmake Lightweight --top 10

# Build a dream card
python main.py dreamcard --fights 5

# Quick demo (Lightweight top 10 + dream card)
python main.py demo
```

## How It Works

1. Scrape fighter stats, fight history, and bonus labels from UFCStats and Wikipedia
2. Engineer **115 features** per matchup — career stats, rolling form, odds, matchup cross-features, fight context
3. RFECV selects the **12 most predictive** features (`models/pipeline_config.py`)
4. A tiny neural network (**257 params**, 12 → 16 → 1) predicts entertainment probability
5. The matchmaker scores every possible pairing in a division and ranks by P(bonus fight)

Inference per pairing:
```
fighter_a + fighter_b
  -> build_full_matchup_vector()    [115-dim]
  -> subset_full_feature_vector()   [12-dim]
  -> StandardScaler.transform()
  -> FightBonusNN.forward() -> sigmoid -> P(bonus fight)
```

Both orderings (A,B) and (B,A) are scored and averaged for symmetry.

## Model Performance

- **AUC = 0.5991** (5-fold temporal CV reference; latest sweep AUC 0.6086 on val)
- **257 parameters** — appropriate capacity for ~338 unique training fights, 26.9% positive rate
- 12 selected features include style clash, recent knockdowns, performance consistency,
  strike trends, and a five-rounder flag

## Project Structure

```
models/
  matchmaker_v2.py         The product: rank matchups by entertainment
  nn_binary.py             Neural network (12 -> 16 -> 1) + training sweep
  feature_engineering.py   115-dim feature vector
  rolling_features.py      Fight-level rolling stats (leak-safe)
  pipeline_config.py       Feature selection + temporal split config
  data_loader.py           Canonical train/val/test pipeline
  data_splits.py           Temporal split + augmentation
  baselines.py             LogReg / RF / XGBoost reference baselines
  feature_selection.py     RFECV pipeline
  xgb_tuning.py            XGBoost hyperparameter sweep
  pca_analysis.py          PCA variance analysis (experimental)
  pca_pipeline.py          PCA-based features (experimental)
  interpretability.py      SHAP analysis
  backtesting.py           Per-event time-respecting backtest
  experiment_summary.py    Tables + plots for the deck
  fight_quality_nn.py      LEGACY 48-dim regression NN
  matchmaker.py            LEGACY heuristic matchmaker
  training.py              LEGACY regression training loop
  checkpoints/
    nn_12feat.pt           Trained 12-feat binary classifier
    scaler_12feat.pkl      StandardScaler (fit on 12-dim train)
scrapers/                  UFCStats + Tapology + Wikipedia scrapers
data/                      SQLite DB + label imports
outputs/                   Plots, experiment logs, backtest reports
dashboard/                 Rich-terminal pretty printers used by main.py
```

## CLI

| Command | What it does |
|---|---|
| `python main.py collect` | Scrape fresh data into SQLite |
| `python main.py train` | Retrain the 12-feat NN; saves `models/checkpoints/nn_12feat.pt` + `scaler_12feat.pkl` |
| `python main.py matchmake Lightweight --top 10` | Rank pairings in a division |
| `python main.py dreamcard --fights 5` | Best fights across all eight men's divisions |
| `python main.py evaluate` | NN AUC / F1 on the held-out test split |
| `python main.py backtest --from 2026-01-01` | Per-event time-respecting backtest |
| `python main.py experiments` | Generate summary tables + plots |
| `python main.py demo` | Lightweight top 10 + 5-fight dream card |
| `python main.py stats` | DB record counts |
| `python main.py seed-db` | Minimal SQLite for offline tests (no scraping) |
| `python main.py pca` | PCA variance breakdown on the full 115-dim feature matrix |

## Team

- **Raji** — Model architecture, baselines, SHAP, integration
- **Lorenzo** — Data labels, scraping, backtesting
- **Gustave** — Cross-features, PCA, code cleanup, dashboard
- **Mattheus** — Temporal splits, CV, calibration, presentation
