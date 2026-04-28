"""
models/feature_engineering.py
Feature extraction for the fight quality Neural Network.

Levels of features:
  1. Fighter features  — career stats, style, physical attributes (24 per fighter)
  2. Matchup features  — cross-fighter signals (24)
  3. Odds features     — betting line signals (5)
  4. Context features  — card/stakes/division (4): title, main event, rounds, weight-class finish proxy
  5. Rolling features — leak-safe recent form / fight_stats aggregates (15 + 15 + 4 matchup)

  → Total input vector: 115 features (81 base + 34 rolling)

Card-position and title flags may partially capture selection bias: the UFC is
more likely to award bonuses on marquee bouts. Compare models with and without
context features to quantify the effect in evaluation write-ups.

Fight Quality Score target is computed from historical fight outcomes.
"""
import logging
import math
import numpy as np
from typing import Optional

from models.rolling_features import (
    ROLLING_DIM_FIGHTER,
    ROLLING_FIGHTER_FEATURE_NAMES,
    ROLLING_MATCHUP_FEATURE_NAMES,
    _career_fallback_vec,
    compute_rolling_matchup_features,
)

logger = logging.getLogger(__name__)

# ── Fighter Feature Schema (24 per fighter) ───────────────────────────────────
FIGHTER_FEATURE_NAMES = [
    # Physical
    "height_cm",          # 0  Raw height
    "reach_cm",           # 1  Raw reach
    "reach_height_ratio", # 2  Reach / height (wingspan proxy)
    # Career
    "total_fights",       # 3
    "win_pct",            # 4
    "finish_rate",        # 5  (KO + Sub) / wins
    "ko_rate",            # 6
    "sub_rate",           # 7
    "dec_rate",           # 8
    # Offense
    "sig_strikes_pm",     # 9  Sig strikes landed per minute
    "sig_strike_acc",     # 10 Strike accuracy
    "td_avg",             # 11 Takedowns per 15 min
    "td_acc",             # 12 TD accuracy
    "sub_avg",            # 13 Sub attempts per 15 min
    # Defense
    "sig_strike_def",     # 14 Sig strike defense %
    "td_def",             # 15 TD defense %
    "sig_strikes_abs_pm", # 16 Sig strikes absorbed per minute
    # Style ratios
    "grapple_ratio",      # 17 TD activity relative to striking
    "ctrl_time_avg",      # 18 Avg control time per fight
    # Durability proxy
    "loss_ko_rate",       # 19 KO losses / total losses
    "loss_sub_rate",      # 20 Sub losses / total losses
    # Activity/momentum
    "fight_frequency",    # 21 Fights per year (last 3 years)
    "recent_form",        # 22 Win rate in last 5 fights (0-1)
    "experience_score",   # 23 log(total_fights + 1) normalized
]

# ── Matchup Feature Schema (24 cross-fighter) ────────────────────────────────
MATCHUP_FEATURE_NAMES = [
    # Physical differential
    "reach_diff_cm",         # 0  Fighter A reach - Fighter B reach
    "height_diff_cm",        # 1
    "size_diff_abs",         # 2  abs(reach + height combined diff)
    # Style clash (high = more contrast = more interesting)
    "style_clash_score",     # 3  |grapple_ratio_A - grapple_ratio_B|
    "striker_vs_grappler",   # 4  sig_strikes_pm_A * td_avg_B (cross product)
    "finisher_clash",        # 5  finish_rate_A * finish_rate_B
    "ko_power_clash",        # 6  ko_rate_A * ko_rate_B (KO vs KO = exciting)
    "sub_threat_clash",      # 7  sub_rate_A * sub_rate_B
    # Offense vs Defense matchup
    "strike_off_vs_def",     # 8  sig_strikes_pm_A / (1 + sig_strike_def_B)
    "td_off_vs_def",         # 9  td_avg_A / (1 + td_def_B)
    "sub_off_vs_def",        # 10 sub_avg_A vs sub_rate_B
    # Competitive balance
    "win_pct_diff",          # 11 |win_pct_A - win_pct_B| (0 = balanced)
    "experience_diff",       # 12 |total_fights_A - total_fights_B|
    "ranking_diff",          # 13 |ranking_A - ranking_B| (0 = closely ranked)
    "finish_rate_sum",       # 14 Both want to finish
    "total_sig_output",      # 15 sig_strikes_pm_A + sig_strikes_pm_B
    "total_td_output",       # 16 td_avg_A + td_avg_B
    # Upset potential
    "upset_potential",       # 17 computed from odds if available, else win_pct_diff
    # Action density prediction
    "predicted_action",      # 18 heuristic: total output × finish rate
    "ground_game_intensity", # 19 (td_avg_A + sub_avg_A + ctrl_time_A) total
    "cardio_test_proxy",     # 20 both have low finish rate → likely go to dec
    # Marketability signals
    "combined_finish_rate",  # 21 avg finish rate
    "title_proximity",       # 22 combined ranking proximity to title (low # = better)
    "form_clash",            # 23 |recent_form_A - recent_form_B| (momentum clash)
]

# ── Odds Feature Schema (5 betting-line signals) ─────────────────────────────
ODDS_FEATURE_NAMES = [
    "odds_closeness",     # 0  1 - |prob_A - prob_B|; 1 = pick'em, 0 = huge mismatch
    "odds_gap",           # 1  prob_A - prob_B (directional; positive = A is favorite)
    "is_close_line",      # 2  Binary: both within ~±200 American odds (gap < 0.2)
    "overround",          # 3  prob_A + prob_B; usually > 1 due to vig
    "has_odds",           # 4  1.0 if real odds data exists, 0.0 if imputed
]

# Proxy for division-level finish rates (lighter divisions → more decisions on average).
WEIGHT_CLASS_FINISH_TENDENCY = {
    "Heavyweight": 0.9,
    "Light Heavyweight": 0.75,
    "Middleweight": 0.65,
    "Welterweight": 0.55,
    "Lightweight": 0.55,
    "Featherweight": 0.55,
    "Bantamweight": 0.5,
    "Flyweight": 0.45,
    "Women's Bantamweight": 0.45,
    "Women's Flyweight": 0.4,
    "Women's Strawweight": 0.4,
}

CONTEXT_FEATURE_NAMES = [
    "is_title_fight",
    "is_main_event",
    "is_five_rounder",
    "weight_class_finish_tendency",
]

ALL_FEATURE_NAMES = (
    [f"f1_{n}" for n in FIGHTER_FEATURE_NAMES]
    + [f"f2_{n}" for n in FIGHTER_FEATURE_NAMES]
    + MATCHUP_FEATURE_NAMES
    + ODDS_FEATURE_NAMES
    + CONTEXT_FEATURE_NAMES
    + [f"f1_roll_{n}" for n in ROLLING_FIGHTER_FEATURE_NAMES]
    + [f"f2_roll_{n}" for n in ROLLING_FIGHTER_FEATURE_NAMES]
    + ROLLING_MATCHUP_FEATURE_NAMES
)

# First block of ALL_FEATURE_NAMES: 24 + 24 + 24 (no odds / context / rolling).
CAREER_CROSS_DIM = len(FIGHTER_FEATURE_NAMES) * 2 + len(MATCHUP_FEATURE_NAMES)
CAREER_CROSS_FEATURE_NAMES = (
    [f"f1_{n}" for n in FIGHTER_FEATURE_NAMES]
    + [f"f2_{n}" for n in FIGHTER_FEATURE_NAMES]
    + list(MATCHUP_FEATURE_NAMES)
)
assert CAREER_CROSS_DIM == 72
assert list(ALL_FEATURE_NAMES[:CAREER_CROSS_DIM]) == list(CAREER_CROSS_FEATURE_NAMES)


def subset_full_feature_vector(
    full_vec: np.ndarray,
    keep: list[str] | None,
    all_names: list[str] | None = None,
) -> np.ndarray:
    """
    Reduce a vector built with ``build_full_matchup_vector`` to ``keep`` columns (order preserved).
    ``keep`` must be a subset of ``all_names`` / ``ALL_FEATURE_NAMES``. None or empty → no-op.
    """
    v = np.asarray(full_vec, dtype=np.float32).ravel()
    if not keep:
        return v
    names = list(all_names) if all_names is not None else list(ALL_FEATURE_NAMES)
    idx = [names.index(n) for n in keep]
    return v[idx]


def extract_fighter_features(fighter: dict) -> np.ndarray:
    """
    Convert a fighter dict (from DB) into a 24-dim feature vector.
    All values are normalized to [0, 1] or centered.
    """
    f = fighter  # alias

    total_fights = (f.get("wins_total") or 0) + (f.get("losses_total") or 0)
    wins = f.get("wins_total") or 0
    losses = f.get("losses_total") or 0
    win_pct = wins / total_fights if total_fights > 0 else 0.5

    # Physical (normalized to typical fighter ranges)
    height_cm = _norm(f.get("height_cm") or 175.0, 155.0, 210.0)
    reach_cm  = _norm(f.get("reach_cm")  or 175.0, 155.0, 215.0)
    reach_height_ratio = (f.get("reach_cm") or 175.0) / max(f.get("height_cm") or 175.0, 1.0)
    reach_height_ratio = _norm(reach_height_ratio, 0.9, 1.1)

    # Career
    total_fights_norm = min(total_fights / 40.0, 1.0)
    finish_rate  = f.get("finish_rate") or 0.0
    ko_rate      = f.get("ko_rate")     or 0.0
    sub_rate     = f.get("sub_rate")    or 0.0
    dec_rate     = f.get("dec_rate")    or 0.0

    # Offense (normalized to typical UFC ranges)
    sig_pm    = _norm(f.get("sig_strikes_pm") or 3.0, 0.0, 12.0)
    sig_acc   = f.get("sig_strike_acc") or 0.44
    td_avg    = _norm(f.get("td_avg") or 1.0, 0.0, 8.0)
    td_acc    = f.get("td_acc") or 0.4
    sub_avg   = _norm(f.get("sub_avg") or 0.5, 0.0, 5.0)

    # Defense
    sig_def   = f.get("sig_strike_def") or 0.5
    td_def    = f.get("td_def") or 0.5
    abs_pm    = _norm(f.get("sig_strikes_abs_pm") or 3.0, 0.0, 12.0)

    # Style
    grapple_ratio = f.get("grapple_ratio") or 0.3
    ctrl_avg      = _norm(f.get("ctrl_time_avg") or 60.0, 0.0, 300.0)

    # Durability
    loss_ko_rate  = (f.get("losses_ko")  or 0) / max(losses, 1)
    loss_sub_rate = (f.get("losses_sub") or 0) / max(losses, 1)

    # Activity — slots 21–22 overwritten when `_rolling_vec` is attached (see below)
    fight_frequency = min(total_fights / 15.0, 1.0)  # fallback if no rolling
    recent_form = win_pct  # fallback if no rolling
    experience_score = min(math.log(total_fights + 1) / math.log(41), 1.0)

    vec = np.array([
        height_cm, reach_cm, reach_height_ratio,
        total_fights_norm, win_pct, finish_rate, ko_rate, sub_rate, dec_rate,
        sig_pm, sig_acc, td_avg, td_acc, sub_avg,
        sig_def, td_def, abs_pm,
        grapple_ratio, ctrl_avg,
        loss_ko_rate, loss_sub_rate,
        fight_frequency, recent_form, experience_score,
    ], dtype=np.float32)

    vec = np.clip(vec, 0.0, 1.0)
    rv = f.get("_rolling_vec")
    if rv is not None:
        rv = np.asarray(rv, dtype=np.float32).ravel()
        if rv.size >= ROLLING_DIM_FIGHTER:
            vec[21] = 1.0 - float(rv[3])
            vec[22] = float(np.clip(rv[0], 0.0, 1.0))
    return vec


def extract_matchup_features(fighter_a: dict, fighter_b: dict) -> np.ndarray:
    """
    Extract 24 cross-fighter matchup features.
    Order is NOT symmetric — caller should average both orderings if needed.
    """
    # Raw values needed for cross products
    sig_pm_a  = fighter_a.get("sig_strikes_pm") or 3.0
    sig_pm_b  = fighter_b.get("sig_strikes_pm") or 3.0
    td_avg_a  = fighter_a.get("td_avg") or 1.0
    td_avg_b  = fighter_b.get("td_avg") or 1.0
    sub_avg_a = fighter_a.get("sub_avg") or 0.5
    sub_avg_b = fighter_b.get("sub_avg") or 0.5
    sig_def_a = fighter_a.get("sig_strike_def") or 0.5
    sig_def_b = fighter_b.get("sig_strike_def") or 0.5
    td_def_a  = fighter_a.get("td_def") or 0.5
    td_def_b  = fighter_b.get("td_def") or 0.5
    grap_a    = fighter_a.get("grapple_ratio") or 0.3
    grap_b    = fighter_b.get("grapple_ratio") or 0.3
    fin_a     = fighter_a.get("finish_rate") or 0.5
    fin_b     = fighter_b.get("finish_rate") or 0.5
    ko_a      = fighter_a.get("ko_rate") or 0.3
    ko_b      = fighter_b.get("ko_rate") or 0.3
    sub_a     = fighter_a.get("sub_rate") or 0.2
    sub_b     = fighter_b.get("sub_rate") or 0.2

    wins_a   = fighter_a.get("wins_total") or 0
    losses_a = fighter_a.get("losses_total") or 0
    wins_b   = fighter_b.get("wins_total") or 0
    losses_b = fighter_b.get("losses_total") or 0
    total_a  = max(wins_a + losses_a, 1)
    total_b  = max(wins_b + losses_b, 1)
    win_pct_a = wins_a / total_a
    win_pct_b = wins_b / total_b
    rank_a    = fighter_a.get("ranking") or 15
    rank_b    = fighter_b.get("ranking") or 15
    height_a  = fighter_a.get("height_cm") or 175.0
    height_b  = fighter_b.get("height_cm") or 175.0
    reach_a   = fighter_a.get("reach_cm") or 175.0
    reach_b   = fighter_b.get("reach_cm") or 175.0
    ctrl_a    = fighter_a.get("ctrl_time_avg") or 60.0

    # Physical
    reach_diff  = _norm_diff(reach_a - reach_b, -20.0, 20.0)
    height_diff = _norm_diff(height_a - height_b, -20.0, 20.0)
    size_diff   = abs(reach_diff) + abs(height_diff)

    # Style clash
    style_clash    = min(abs(grap_a - grap_b) * 2.0, 1.0)
    striker_grap   = _norm(sig_pm_a * td_avg_b + sig_pm_b * td_avg_a, 0, 50)
    finisher_clash = fin_a * fin_b
    ko_clash       = ko_a * ko_b
    sub_clash      = sub_a * sub_b

    # Offense vs defense
    strike_off_def = _norm((sig_pm_a / max(1 - sig_def_b, 0.01)), 0, 15)
    td_off_def     = _norm((td_avg_a / max(1 - td_def_b, 0.01)), 0, 15)
    sub_off_def    = sub_avg_a * (1 - sub_b)

    # Competitive balance
    win_pct_diff   = abs(win_pct_a - win_pct_b)
    exp_diff       = abs(total_a - total_b) / 40.0
    rank_diff      = abs(rank_a - rank_b) / 15.0
    finish_sum     = _norm(fin_a + fin_b, 0, 2.0)

    # Action density prediction
    total_output   = _norm(sig_pm_a + sig_pm_b, 0, 20)
    total_td       = _norm(td_avg_a + td_avg_b, 0, 12)
    upset_pot      = win_pct_diff  # higher difference = more upset potential
    pred_action    = _norm((sig_pm_a + sig_pm_b) * (fin_a + fin_b) / 2, 0, 20)
    ground_intens  = _norm(td_avg_a + sub_avg_a + ctrl_a / 60 + td_avg_b + sub_avg_b, 0, 20)
    cardio_test    = (1 - fin_a) * (1 - fin_b)

    # Marketability
    combined_finish = (fin_a + fin_b) / 2.0
    title_prox      = _norm((rank_a + rank_b) / 2, 0, 15)
    form_clash      = abs(win_pct_a - win_pct_b)

    vec = np.array([
        reach_diff, height_diff, size_diff,
        style_clash, striker_grap, finisher_clash, ko_clash, sub_clash,
        strike_off_def, td_off_def, sub_off_def,
        win_pct_diff, exp_diff, rank_diff, finish_sum,
        total_output, total_td,
        upset_pot, pred_action, ground_intens, cardio_test,
        combined_finish, title_prox, form_clash,
    ], dtype=np.float32)

    return np.clip(vec, 0.0, 1.0)


def extract_odds_features(fighter_a_odds, fighter_b_odds) -> np.ndarray:
    """
    Extract 5 betting-line features from American odds.

    Parameters
    ----------
    fighter_a_odds, fighter_b_odds : numeric American odds (e.g. -150, +130)
        or None/0 when unavailable.

    Returns 5-dim vector: [odds_closeness, odds_gap, is_close_line, overround, has_odds]
    """
    has_real_odds = (
        fighter_a_odds is not None and fighter_b_odds is not None
        and fighter_a_odds != 0 and fighter_b_odds != 0
    )

    prob_a = _american_to_prob(fighter_a_odds)
    prob_b = _american_to_prob(fighter_b_odds)

    odds_closeness = 1.0 - abs(prob_a - prob_b)
    odds_gap = prob_a - prob_b
    is_close_line = 1.0 if abs(prob_a - prob_b) < 0.2 else 0.0
    overround = prob_a + prob_b
    has_odds = 1.0 if has_real_odds else 0.0

    return np.array(
        [odds_closeness, odds_gap, is_close_line, overround, has_odds],
        dtype=np.float32,
    )


def make_hypothetical_fight_context(
    *,
    is_title_fight: bool = False,
    is_main_event: bool = False,
    scheduled_rounds: int | None = None,
    weight_class: str = "",
) -> dict:
    """
    Build a fights-table-style dict for extract_context_features().

    Used when scoring hypothetical matchups (no DB fight row). If
    ``scheduled_rounds`` is omitted, uses 5 when ``is_title_fight`` else 3.
    """
    if scheduled_rounds is None:
        scheduled_rounds = 5 if is_title_fight else 3
    return {
        "is_title_fight": int(bool(is_title_fight)),
        "is_main_event": int(bool(is_main_event)),
        "scheduled_rounds": int(scheduled_rounds),
        "weight_class": weight_class or "",
    }


def extract_context_features(fight_data: dict) -> np.ndarray:
    """
    Card / stakes context from the fights table (or hypothetical context dict).

    Uses ``is_title_fight`` and ``is_main_event`` from the DB (headliner flag).
    ``scheduled_rounds`` defaults to 5 if title fight else 3 when missing.

    Note: These signals may correlate with label bias — marquee bouts get more
    bonus consideration from the UFC. See project write-up for ablation discussion.
    """
    fd = fight_data or {}
    is_title = 1.0 if fd.get("is_title_fight") else 0.0
    is_main_evt = 1.0 if fd.get("is_main_event") else 0.0

    sr = fd.get("scheduled_rounds")
    if sr is None:
        scheduled_rounds = 5 if fd.get("is_title_fight") else 3
    else:
        try:
            scheduled_rounds = int(sr)
        except (TypeError, ValueError):
            scheduled_rounds = 3
    is_five = 1.0 if scheduled_rounds == 5 else 0.0

    wc = (fd.get("weight_class") or "").strip()
    finish_tend = WEIGHT_CLASS_FINISH_TENDENCY.get(wc, 0.55)

    return np.array(
        [is_title, is_main_evt, is_five, float(finish_tend)],
        dtype=np.float32,
    )


def build_matchup_vector(fighter_a: dict, fighter_b: dict) -> np.ndarray:
    """
    Build a 48-dim input vector for the legacy regression NN.
    = fighter_a features (24) + fighter_b features (24).

    For the binary classifier pipeline, use build_full_matchup_vector().
    """
    fa = extract_fighter_features(fighter_a)
    fb = extract_fighter_features(fighter_b)
    return np.concatenate([fa, fb])


def build_career_cross_matchup_vector(fighter_a: dict, fighter_b: dict) -> np.ndarray:
    """
    72-dim vector: fighter A (24) + fighter B (24) + matchup cross-features (24).

    No odds, card context, or rolling stats — use for small-sample feature selection
    and ablations (curse of dimensionality with ~300 fights).
    """
    fa = extract_fighter_features(fighter_a)
    fb = extract_fighter_features(fighter_b)
    cross = extract_matchup_features(fighter_a, fighter_b)
    return np.concatenate([fa, fb, cross])


def _rolling_vec_for_fighter(fighter: dict) -> np.ndarray:
    v = fighter.get("_rolling_vec")
    if v is not None:
        out = np.asarray(v, dtype=np.float32).ravel()
        if out.size >= ROLLING_DIM_FIGHTER:
            return out[:ROLLING_DIM_FIGHTER]
    return _career_fallback_vec(fighter)


def build_full_matchup_vector(fighter_a: dict, fighter_b: dict) -> np.ndarray:
    """
    Build the full 115-dim input vector for the binary classification pipeline.

    Layout: fighter_a (24) + fighter_b (24) + matchup cross (24) + odds (5) + context (4)
    + rolling_a (15) + rolling_b (15) + rolling_matchup (4).

    Odds: ``_fight_odds`` per fighter (from build_raw_pairs). Missing → imputed; has_odds=0.

    Context: ``_fight_context`` dict (same on both corners for a real fight). Missing →
    generic defaults (non-title, 3 rounds, default finish tendency). For hypotheticals,
    use make_hypothetical_fight_context() and attach to both fighter dict copies.

    Rolling: ``_rolling_vec`` from leak-safe DB history (training) or
    ``get_inference_rolling_vector`` (matchmaking). If absent → career fallback.
    """
    fa = extract_fighter_features(fighter_a)
    fb = extract_fighter_features(fighter_b)
    cross = extract_matchup_features(fighter_a, fighter_b)
    odds = extract_odds_features(
        fighter_a.get("_fight_odds"),
        fighter_b.get("_fight_odds"),
    )
    ctx_src = fighter_a.get("_fight_context") or fighter_b.get("_fight_context") or {}
    context = extract_context_features(ctx_src if isinstance(ctx_src, dict) else {})
    r_a = _rolling_vec_for_fighter(fighter_a)
    r_b = _rolling_vec_for_fighter(fighter_b)
    r_x = compute_rolling_matchup_features(r_a, r_b)
    return np.concatenate([fa, fb, cross, odds, context, r_a, r_b, r_x])


def compute_fight_quality_score(fight: dict, db) -> Optional[float]:
    """
    Compute a fight quality score (0–100) from observed fight stats.
    This is the TRAINING TARGET for the NN.

    Components:
      - Action density     (sig strikes per min, normalized)
      - Finish drama       (was it finished? + late finish bonus)
      - Competitive balance(how close was the action split)
      - Ground game drama  (TDs + submissions attempted)
      - Knockdown drama    (knockdowns happened = excitement)
    """
    if not fight:
        return None

    total_time = fight.get("total_time_sec") or 0
    if total_time < 30:  # less than 30s is not useful
        return None

    total_sig   = fight.get("total_sig_strikes") or 0
    sig_pm      = fight.get("sig_strikes_pm") or 0.0
    total_tds   = fight.get("total_tds") or 0
    knockdowns  = fight.get("knockdowns") or 0
    method      = (fight.get("method") or "").lower()
    rnd         = fight.get("round") or 1
    total_rounds = 3  # most fights are 3 rounds; title = 5

    fight_stats = db.get_fight_stats(fight["id"])

    # ── Action density (0-30 pts) ─────────────────────────────────────────
    # Elite fight = ~8+ sig strikes/min (e.g. Poirier vs. Gaethje)
    action_score = min(sig_pm / 8.0, 1.0) * 30.0

    # ── Finish drama (0-25 pts) ───────────────────────────────────────────
    is_finish = any(m in method for m in ["ko", "tko", "submission"])
    finish_score = 0.0
    if is_finish:
        # Later-round finishes are more dramatic
        round_bonus = rnd / max(total_rounds, 1)
        finish_score = (0.6 + 0.4 * round_bonus) * 25.0
    else:
        # Decisions can still be good fights — base 8 pts
        finish_score = 8.0

    # ── Competitive balance (0-20 pts) ────────────────────────────────────
    # Closer the action split, better the fight
    balance_score = 20.0
    if fight_stats and len(fight_stats) == 2:
        f1_sig = fight_stats[0].get("sig_strikes_landed") or 0
        f2_sig = fight_stats[1].get("sig_strikes_landed") or 0
        total_in_fight = f1_sig + f2_sig
        if total_in_fight > 0:
            share = max(f1_sig, f2_sig) / total_in_fight
            # share of 0.5 = perfectly balanced → 1.0 score
            # share of 1.0 = one-sided → 0.0 score
            balance_score = (1.0 - (share - 0.5) * 2.0) * 20.0

    # ── Ground game drama (0-15 pts) ─────────────────────────────────────
    sub_attempts = sum(s.get("sub_attempts") or 0 for s in fight_stats) if fight_stats else 0
    ground_score = min((total_tds * 1.5 + sub_attempts * 2.0) / 10.0, 1.0) * 15.0

    # ── Knockdown drama (0-10 pts) ────────────────────────────────────────
    kd_score = min(knockdowns * 3.0, 1.0) * 10.0

    total = action_score + finish_score + balance_score + ground_score + kd_score
    return round(min(total, 100.0), 2)


def compute_fighter_style_metrics(fighter: dict, db) -> dict:
    """
    Compute style metrics from a fighter's historical fight stats.
    These update the DB fighter record.
    """
    fighter_id = fighter.get("id")
    if not fighter_id:
        return {}

    # Get all fight stats for this fighter
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM fight_stats WHERE fighter_id=?", (fighter_id,)
        ).fetchall()

    if not rows:
        return {}

    rows = [dict(r) for r in rows]
    n = len(rows)

    avg_sig = sum(r.get("sig_strikes_landed") or 0 for r in rows) / n
    avg_td  = sum(r.get("td_landed") or 0 for r in rows) / n
    avg_ctrl = sum(r.get("ctrl_time_sec") or 0 for r in rows) / n

    # Grapple ratio: how much of a fighter's output is grappling vs striking
    # (normalized so 0 = pure striker, 1 = pure grappler)
    total_activity = avg_sig + avg_td * 5  # weight TDs more
    grapple_ratio = (avg_td * 5) / max(total_activity, 1.0)

    return {
        "ctrl_time_avg": avg_ctrl,
        "grapple_ratio": round(grapple_ratio, 3),
    }


# ── Utility ──────────────────────────────────────────────────────────────────

def _american_to_prob(odds) -> float:
    """Convert American odds to implied probability (0-1). Unknown → 0.5."""
    if odds is None or odds == 0:
        return 0.5
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def _norm(val: float, lo: float, hi: float) -> float:
    """Normalize to [0, 1]."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def _norm_diff(val: float, lo: float, hi: float) -> float:
    """Normalize a difference to [0, 1] centered at 0.5."""
    return _norm(val, lo, hi)
