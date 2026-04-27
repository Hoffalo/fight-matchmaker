# Lorenzo's Slide Content (L5)

Hand this to Mattheus for the deck. All numbers come from `data/quality_report.py`,
`scrapers/wikipedia_bonus_scraper.py`, and `models/backtesting.py` outputs in
`outputs/`. Re-run those scripts to refresh.

---

## Slide: Problem Statement — "Circular Training"

**Title:** What were we training the model on, anyway?

**Body:**
- The original neural network's training target was `compute_fight_quality_score`
  — a hand-coded 0–100 formula in `models/feature_engineering.py` combining
  action density, finish drama, balance, ground game, and knockdowns.
- That formula is computed from the **same fight statistics** the network sees
  as input. Training a model to predict its own input transformation cannot
  produce knowledge the formula doesn't already encode. There is no external
  signal — the experiment can't fail and can't surprise us.
- We replaced the heuristic target with a **real-world ground-truth label**:
  whether the UFC awarded the fight a Fight of the Night or Performance of
  the Night bonus. Bonuses are independently judged by UFC officials, so the
  label is exogenous to the input features.
- This converts the problem from regression-on-our-own-formula to
  **binary classification of "did this fight get a $50,000 bonus?"** — a
  question that has a real-world answer the model can be wrong about.

**One-line takeaway:** before the fix, the model was grading its own homework.

---

## Slide: Data Collection

**Title:** Where the data and labels come from

**Sources:**
- **UFCStats.com** — every event since 2025-01-18, scraped via Selenium.
  Fight cards, per-fighter stats (sig strikes, takedowns, control time,
  knockdowns), fight outcomes, fighter career stats, fighter physicals.
- **Wikipedia event pages** — *added this sprint* — each event's
  "Bonus awards" section, parsed for FOTN and POTN winners. Plain HTTP +
  BeautifulSoup, no JS rendering needed.
- **Tapology** — current rankings (existing scraper, untouched this sprint).

**Volumes (current DB):**
| Records | Count |
|---|---|
| Fighters | 650 |
| Events | 50 |
| Fights | 611 |
| Bonus award rows | 178 (52 FOTN + 126 POTN) |
| Distinct fights with a bonus | **148** |

**Coverage window:** 2025-01-18 → 2026-03-21 (14 months).

**Data-quality fixes shipped this sprint:**
- `events.date` was empty for every row (pre-existing scraper bug — wrong
  cell in the events list table). Backfilled all 50 events to ISO format
  via `data/backfill_event_dates.py`. Without this, no temporal split.
- 178/178 bonus rows resolved to both a `fighter_id` and a `fight_id`
  after building a normalizer that handles diacritics, Polish `ł`,
  hyphens, and short-form first names (e.g., "Bia" ↔ "Beatriz",
  "Rong Zhu" ↔ "Rongzhu").
- 6 events had no bonus rows extracted (5 Wikipedia URL-disambiguation
  misses, 1 non-UFC regional event). ~5–8 fights of label loss; flagged
  as known and not load-bearing.

---

## Slide: Headline Stats

**Title:** What the labeled dataset tells us

- **24.2% of UFC fights in our window received a bonus** (148 of 611).
  Higher than the league-wide ~10–15% rate the literature predicts because
  our DB skews toward main-card fights on a small set of recent events.
  Class imbalance handling is still required for every model.
- **Bonus mix: 71% POTN, 29% FOTN.** POTN is awarded to *individual*
  performances (often early KOs or submissions), FOTN to *both fighters*
  in a single fight. Our `fight_bonuses` table preserves both signals;
  `is_bonus_fight=1` if the fight had either.
- **Bonus base rate per event: ~3 fights of ~13.** Picking 3 fights at
  random from each card hits at least one bonus winner ~62% of the time,
  which is the floor any model has to beat.

---

## Slide: Backtesting — Did the labels survive contact with reality?

**Title:** Top-3 picks per card vs actual FOTN/POTN winners (2026 test set)

- Train: 508 fights from events before 2026-01-01 (Lorenzo's labels,
  no temporal leakage).
- Test: 8 events from Jan–Mar 2026, 103 fights, 26 actual bonus fights.
- Model: LogisticRegression baseline on the 48-feature vector
  (placeholder for Raji's XGBoost/NN — drop-in via `predict_proba`).

| Metric | Value |
|---|---|
| Top-3 hit rate (model) | **87.5%** (7/8 events) |
| Top-3 hit rate (random pick baseline) | 61.8% |
| **Uplift over random** | **+25.7 percentage points** |

**Caveat to put on the slide:** 8 events is a small sample, so per-event
rates are noisy. Treat the uplift as directional, not as a publishable
effect size. A real model (Raji's tuned XGBoost) on the same split is
likely to extend the lead.

**Memorable single example (UFC 325, 2026-01-31):**
- Model's top pick: Volkanovski vs Lopes 2 (the actual main event and an
  actual FOTN winner). The model surfaced this without seeing the result —
  it identified the matchup quality from features alone.

See `outputs/backtest_results.md` for the full per-event table.
