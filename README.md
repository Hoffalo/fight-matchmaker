# UFC Matchmaker — AI-Powered Fight Pairing System

A full-stack Python system that scrapes UFC fighter and fight data, trains a Neural Network on fight quality metrics, and predicts the best fighter pairings from a business/entertainment perspective.

---

## Architecture

```
ufc_matchmaker/
├── scrapers/
│   ├── sherdog_scraper.py      # Selenium scraper for Sherdog fighter/event data
│   ├── ufc_stats_scraper.py    # Selenium scraper for UFCStats.com (detailed per-fight stats)
│   ├── tapology_scraper.py     # Selenium scraper for Tapology (odds, rankings)
│   └── ufc_api_wrapper.py      # Wrapper around FritzCapuyan/ufc-api
├── data/
│   ├── db.py                   # SQLite database manager
│   ├── schema.sql              # Database schema
│   └── pipeline.py             # Full data collection pipeline
├── models/
│   ├── feature_engineering.py  # Feature extraction from raw data
│   ├── fight_quality_nn.py     # Neural Network: fight quality scorer
│   ├── matchmaker.py           # Matchmaking engine using trained NN
│   └── training.py             # Training loop + evaluation
├── utils/
│   ├── driver.py               # Selenium WebDriver factory
│   ├── rate_limiter.py         # Polite scraping rate limiter
│   └── logger.py               # Structured logging
├── dashboard/
│   └── app.py                  # Rich terminal dashboard (rich + typer CLI)
├── main.py                     # Entry point
├── config.py                   # Configuration
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data (runs all scrapers, populates SQLite DB)
python main.py collect --events 50 --fighters 300

# 3. Train the Neural Network
python main.py train --epochs 100

# 4. Generate fight pairings for a weight class
python main.py predict --weight-class Lightweight --top 20

# 5. Launch interactive dashboard
python main.py dashboard
```

---

## Data Sources

| Source | Data | Method |
|---|---|---|
| UFCStats.com | Per-fight strike/TD/control splits, round-by-round | Selenium |
| Sherdog.com | Fighter career stats, win methods, history | ufc-api + Selenium |
| Tapology.com | Betting odds history, rankings, fan interest | Selenium |
| UFC.com | Official rankings, weight classes | Selenium |

---

## Neural Network

The NN predicts a **Fight Quality Score (0-100)** based on:

**Fighter Features (per fighter):**
- Physical: height, reach, weight, reach advantage
- Offense: sig strike accuracy, KO power (KO rate), submission rate
- Defense: strike defense %, TD defense %
- Style: grappling ratio, clinch ratio, distance ratio
- Activity: fight frequency, recent form (last 5 fights)
- Career: total fights, win %, finish rate

**Matchup Features (cross-referenced):**
- Style clash score (striker vs grappler differential)
- Reach differential
- Size differential
- Finish probability (combined)
- Competitive balance (ranking proximity)
- Crowd appeal proxy (finish rate × ranking × opponent quality)

**Business Outcome Targets (from historical data):**
- Fight duration score (longer = generally better for fans)
- Action density (sig strikes per minute)
- Finish drama score (late finish > early finish for drama)
- Upset potential (odds differential)
- Rematch demand proxy

---

## Neural Network Architecture

```
INPUT
┌─────────────────────────────────────────────────────────────────┐
│  Feature Vector  [batch, 48]                                    │
│  48 floats: 24 per fighter (physical, offense, defense,         │
│  style, activity, career) + matchup cross-features              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    ── ENCODER ──
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Linear(48 → 256)                                               │
│  Expands the input into a rich 256-dim representation.          │
│  Gives the network enough capacity to learn complex feature     │
│  interactions between the two fighters.                         │
│  + LayerNorm(256)  — normalizes activations for stable training │
│  + GELU            — smooth non-linearity (better than ReLU     │
│                       for deep nets; allows small neg. values)  │
│  + Dropout(0.3)    — randomly zeros 30% of neurons to prevent  │
│                       overfitting on limited fight data         │
└────────────────────────┬────────────────────────────────────────┘
                         │ [batch, 256]
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Linear(256 → 128)                                              │
│  First compression step. Forces the network to distill the      │
│  256-dim representation into the 128 most predictive features.  │
│  + LayerNorm(128)                                               │
│  + GELU                                                         │
│  + Dropout(0.3)                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ [batch, 128]
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Linear(128 → 64)                                               │
│  Second compression. Abstracts fighter matchup patterns into    │
│  a compact 64-dim space (e.g. "striker vs grappler",            │
│  "ranked vs unranked", "brawler vs technician").                │
│  + LayerNorm(64)                                                │
│  + GELU                                                         │
│  + Dropout(0.3)                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ [batch, 64]
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Linear(64 → 32)                                                │
│  Final encoder layer. No Dropout here — the 32-dim bottleneck  │
│  is the learned "fight quality embedding" passed to the head.   │
│  + LayerNorm(32)                                                │
│  + GELU                                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │ [batch, 32]
                    ── HEAD ──
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Linear(32 → 16)                                                │
│  Small projection that re-weights the fight quality embedding   │
│  before the final scalar prediction.                            │
│  + GELU                                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │ [batch, 16]
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Linear(16 → 1)                                                 │
│  Collapses all features into a single raw score.                │
│  + Sigmoid  — squashes output to (0, 1)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │ [batch, 1]  ∈ (0, 1)
                         ▼
                   × 100  (post-hoc)
                         │
                         ▼
                  Fight Quality Score
                     0 — 100
```

| Layer | Input | Output | Purpose |
|---|---|---|---|
| Linear 1 | 48 | 256 | Expand features, learn pairwise interactions |
| Linear 2 | 256 | 128 | Compress, extract dominant signals |
| Linear 3 | 128 | 64 | Abstract matchup archetypes |
| Linear 4 | 64 | 32 | Bottleneck fight quality embedding |
| Linear 5 | 32 | 16 | Re-weight embedding for final prediction |
| Linear 6 | 16 | 1 | Scalar score (sigmoid → ×100) |

All encoder layers use **LayerNorm + GELU + Dropout(0.3)** except the last encoder layer which omits Dropout.
Weights are initialized with **Kaiming Normal** (suited for GELU/ReLU activations).

---

## Fight Quality Scoring

```
FightQualityScore = w1*ActionDensity + w2*FinishProbability + 
                    w3*CompetitiveBalance + w4*StyleClash + 
                    w5*MarketabilityScore + w6*UpsetPotential
```

Weights learned by the NN from historical fight data.

---

## Output Example

```
TOP PREDICTED MATCHUPS — Lightweight Division
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#1  Islam Makhachev   vs  Charles Oliveira     Score: 94.2 / 100
    Style Clash: ★★★★★  Balance: ★★★★★  Finish Prob: 67%
    "Elite grappler vs elite submission artist — action guaranteed"

#2  Dustin Poirier    vs  Justin Gaethje       Score: 91.8 / 100
    Style Clash: ★★★★☆  Balance: ★★★★★  Finish Prob: 78%
    "High-output strikers, both durable — likely war"
```
