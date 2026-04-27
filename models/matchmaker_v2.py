"""
models/matchmaker_v2.py
Entertainment-optimised matchmaking engine (v2).

Uses a trained binary classifier (XGBoost, Random Forest, or FightBonusNN) to
score all possible fighter pairings by P(bonus fight), then ranks them to find
the matchups most likely to produce entertaining fights.

Pipeline per pairing:
  1. Look up both fighters in the DB
  2. extract_fighter_features(A) → 24 dims
  3. extract_fighter_features(B) → 24 dims
  4. extract_matchup_features(A, B) → 24 dims
  5. Concatenate → 72-dim vector
  6. Average both orderings (A,B) and (B,A) for symmetry
  7. Scale with the fitted StandardScaler
  8. Model.predict_proba → P(bonus fight)

The greedy card builder reuses the constraint logic from matchmaker.py:
each fighter appears at most once on a card.
"""
import logging
import random
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Star rating mapping
# ─────────────────────────────────────────────────────────────────────────────

def star_rating(probability: float) -> int:
    """Map entertainment probability to a 1–5 star rating."""
    if probability >= 0.75:
        return 5
    if probability >= 0.55:
        return 4
    if probability >= 0.40:
        return 3
    if probability >= 0.25:
        return 2
    return 1


def star_string(stars: int) -> str:
    return "\u2605" * stars + "\u2606" * (5 - stars)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredMatchup:
    """Result of scoring a single fighter pairing."""
    fighter_a: str
    fighter_b: str
    fighter_a_id: int
    fighter_b_id: int
    entertainment_probability: float
    entertainment_rating: int
    top_factors: list[str]
    style_summary: str
    raw_features: dict = field(default_factory=dict, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder (wraps feature_engineering.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_72_vector(fighter_a: dict, fighter_b: dict, fe_module) -> np.ndarray:
    """
    Build the full 72-dim feature vector from two fighter stat dicts.

    Layout: [fighter_A_24 | fighter_B_24 | matchup_cross_24]
    """
    fa = fe_module.extract_fighter_features(fighter_a)     # (24,)
    fb = fe_module.extract_fighter_features(fighter_b)     # (24,)
    cross = fe_module.extract_matchup_features(fighter_a, fighter_b)  # (24,)
    return np.concatenate([fa, fb, cross])


def _build_symmetric_vector(fighter_a: dict, fighter_b: dict, fe_module) -> np.ndarray:
    """Average both orderings so A-vs-B == B-vs-A."""
    vec_ab = _build_72_vector(fighter_a, fighter_b, fe_module)
    vec_ba = _build_72_vector(fighter_b, fighter_a, fe_module)
    return ((vec_ab + vec_ba) / 2.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Style summary generator
# ─────────────────────────────────────────────────────────────────────────────

def _style_summary(a: dict, b: dict) -> str:
    """One-line stylistic description of the matchup."""
    gr_a = a.get("grapple_ratio") or 0.3
    gr_b = b.get("grapple_ratio") or 0.3
    fin_a = a.get("finish_rate") or 0.5
    fin_b = b.get("finish_rate") or 0.5
    spm_a = a.get("sig_strikes_pm") or 3.0
    spm_b = b.get("sig_strikes_pm") or 3.0

    if abs(gr_a - gr_b) > 0.3:
        striker = a.get("name", "A") if gr_a < gr_b else b.get("name", "B")
        grappler = b.get("name", "B") if gr_a < gr_b else a.get("name", "A")
        return f"Striker vs Grappler ({striker} striking, {grappler} grappling) — high clash potential"
    if gr_a > 0.5 and gr_b > 0.5:
        return "Grappling chess match — submission and control threats both ways"
    if spm_a > 5.0 and spm_b > 5.0:
        return "High-volume striking war — fireworks expected"
    if fin_a > 0.6 and fin_b > 0.6:
        return "Two elite finishers — someone is getting stopped"
    if spm_a + spm_b > 8.0:
        return "Action-heavy matchup with combined high striking output"
    return "Competitive all-rounders with overlapping skill sets"


def _top_factors(a: dict, b: dict) -> list[str]:
    """Derive the top 3 reasons this matchup should be entertaining."""
    factors: list[tuple[float, str]] = []

    fin_a = a.get("finish_rate") or 0.5
    fin_b = b.get("finish_rate") or 0.5
    combined_finish = (fin_a + fin_b) / 2
    factors.append((combined_finish, f"Combined finish rate: {combined_finish:.0%}"))

    gr_a = a.get("grapple_ratio") or 0.3
    gr_b = b.get("grapple_ratio") or 0.3
    clash = abs(gr_a - gr_b)
    if clash > 0.15:
        factors.append((clash, f"Style clash score: {clash:.2f} (striker vs grappler dynamic)"))

    spm_a = a.get("sig_strikes_pm") or 3.0
    spm_b = b.get("sig_strikes_pm") or 3.0
    pace = (spm_a + spm_b) / 12.0
    factors.append((pace, f"Projected pace: {spm_a + spm_b:.1f} sig. strikes/min combined"))

    wp_a = a.get("wins_total", 0) / max(a.get("wins_total", 0) + a.get("losses_total", 0), 1)
    wp_b = b.get("wins_total", 0) / max(b.get("wins_total", 0) + b.get("losses_total", 0), 1)
    balance = 1.0 - abs(wp_a - wp_b)
    factors.append((balance, f"Competitive balance: {balance:.2f}"))

    ko_a = a.get("ko_rate") or 0.3
    ko_b = b.get("ko_rate") or 0.3
    ko_clash = ko_a * ko_b
    if ko_clash > 0.1:
        factors.append((ko_clash + 0.3, f"KO power both ways (KO rates: {ko_a:.0%} vs {ko_b:.0%})"))

    rank_a = a.get("ranking")
    rank_b = b.get("ranking")
    if rank_a and rank_b and rank_a <= 5 and rank_b <= 5:
        factors.append((1.0, f"Title eliminator: #{rank_a} vs #{rank_b}"))

    factors.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in factors[:3]]


# ─────────────────────────────────────────────────────────────────────────────
# Narrative generator
# ─────────────────────────────────────────────────────────────────────────────

_OPENING_TEMPLATES = [
    "{a} vs {b} projects as a {stars}-star matchup ({prob:.0%} entertainment probability).",
    "The {a}–{b} pairing grades out at {prob:.0%} on the entertainment model — {stars} stars.",
    "Model gives {a} vs {b} a {prob:.0%} entertainment score ({stars}-star rating).",
    "At {prob:.0%}, {a} vs {b} is a {stars}-star projected bout.",
]

_FINISH_TEMPLATES = [
    "Both fighters have elite finishing ability (combined finish rate: {rate:.0%}), making an early stoppage likely.",
    "With a combined {rate:.0%} finish rate, this one probably doesn't go the distance.",
    "The combined finishing pedigree ({rate:.0%}) suggests high drama and a probable stoppage.",
]

_CLASH_TEMPLATES = [
    "The {archetype_a}-vs-{archetype_b} dynamic creates high-variance exchanges where anything can happen.",
    "A classic style clash — {name_a}'s {archetype_a} approach meets {name_b}'s {archetype_b} game plan.",
    "Stylistically these two are polar opposites, which historically produces the most entertaining fights.",
]

_PACE_TEMPLATES = [
    "Both fighters push a relentless pace ({total_spm:.1f} combined sig. strikes per minute).",
    "Expect non-stop action with {total_spm:.1f} combined significant strikes per minute on average.",
]

_BALANCE_TEMPLATES = [
    "The competitive balance score is tight ({balance:.2f}), suggesting neither fighter has a dominant advantage.",
    "On paper this is a coin-flip ({balance:.2f} balance) — and those fights tend to deliver.",
    "Neither fighter is a clear favourite (balance: {balance:.2f}), which adds to the drama.",
]


def _generate_narrative(
    a: dict,
    b: dict,
    prob: float,
    stars: int,
) -> str:
    """Generate a varied, data-driven narrative for a matchup."""
    rng = random.Random(hash((a.get("name", ""), b.get("name", ""))))
    parts: list[str] = []

    name_a = a.get("name", "Fighter A")
    name_b = b.get("name", "Fighter B")

    # Opening
    tmpl = rng.choice(_OPENING_TEMPLATES)
    parts.append(tmpl.format(a=name_a, b=name_b, prob=prob, stars=stars))

    # Finishing ability
    fin_a = a.get("finish_rate") or 0.5
    fin_b = b.get("finish_rate") or 0.5
    combined_finish = (fin_a + fin_b) / 2
    if combined_finish > 0.45:
        parts.append(rng.choice(_FINISH_TEMPLATES).format(rate=combined_finish))

    # Style clash
    gr_a = a.get("grapple_ratio") or 0.3
    gr_b = b.get("grapple_ratio") or 0.3
    if abs(gr_a - gr_b) > 0.2:
        arch_a = "striking" if gr_a < gr_b else "grappling"
        arch_b = "grappling" if gr_a < gr_b else "striking"
        parts.append(rng.choice(_CLASH_TEMPLATES).format(
            archetype_a=arch_a, archetype_b=arch_b,
            name_a=name_a, name_b=name_b,
        ))

    # Pace
    spm_a = a.get("sig_strikes_pm") or 3.0
    spm_b = b.get("sig_strikes_pm") or 3.0
    total_spm = spm_a + spm_b
    if total_spm > 7.0:
        parts.append(rng.choice(_PACE_TEMPLATES).format(total_spm=total_spm))

    # Balance
    wp_a = a.get("wins_total", 0) / max(a.get("wins_total", 0) + a.get("losses_total", 0), 1)
    wp_b = b.get("wins_total", 0) / max(b.get("wins_total", 0) + b.get("losses_total", 0), 1)
    balance = 1.0 - abs(wp_a - wp_b)
    if balance > 0.7:
        parts.append(rng.choice(_BALANCE_TEMPLATES).format(balance=balance))

    # Title implications
    rank_a = a.get("ranking")
    rank_b = b.get("ranking")
    if rank_a and rank_b and rank_a <= 5 and rank_b <= 5:
        parts.append(f"With both fighters ranked in the top 5 (#{rank_a} vs #{rank_b}), "
                      "this carries title-eliminator weight.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Model adapter — unified interface for sklearn-like and NN models
# ─────────────────────────────────────────────────────────────────────────────

def _model_predict_proba(
    model: Any,
    X: np.ndarray,
    scaler: Optional[StandardScaler],
) -> np.ndarray:
    """
    Return P(positive class) for each row in X.

    Supports:
      - sklearn / xgboost classifiers  (have .predict_proba method)
      - FightBonusNN via nn_binary.predict_proba()
      - Any callable that takes (model, X, scaler) and returns probs
    """
    X_input = scaler.transform(X) if scaler is not None else X

    if hasattr(model, "predict_proba") and callable(model.predict_proba):
        return model.predict_proba(X_input)[:, 1]

    # PyTorch NN path — import lazily to avoid hard torch dependency
    try:
        from models.nn_binary import predict_proba as nn_predict_proba
        return nn_predict_proba(model, X_input, scaler=None)
    except ImportError:
        pass

    raise TypeError(
        f"Model type {type(model).__name__} is not supported. "
        "Expected an sklearn-like classifier or a FightBonusNN."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MatchmakerV2
# ─────────────────────────────────────────────────────────────────────────────

class MatchmakerV2:
    """
    Entertainment-optimised matchmaking engine.

    Scores all possible fighter pairings using a trained binary classifier,
    then ranks them to surface the matchups most likely to produce
    bonus-worthy fights.

    Usage
    -----
        from models.matchmaker_v2 import MatchmakerV2
        import models.feature_engineering as fe

        mm = MatchmakerV2(model, scaler, fe, db_path="data/ufc_matchmaker.db")
        top10 = mm.rank_weight_class("Lightweight", top_n=10)
        card  = mm.build_card(["Lightweight", "Welterweight"], total_fights=5)
    """

    def __init__(
        self,
        model: Any,
        scaler: Optional[StandardScaler],
        feature_engineering_module: Any,
        db_path: Optional[str] = None,
        fighter_cache: Optional[dict[int, dict]] = None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.fe = feature_engineering_module
        self.db_path = db_path
        self._fighter_cache: dict[int, dict] = fighter_cache or {}

    # ── DB helpers ────────────────────────────────────────────────────────

    def _get_db(self):
        from data.db import Database
        return Database(self.db_path)

    def _get_fighter(self, fighter_id: int) -> dict:
        if fighter_id in self._fighter_cache:
            return self._fighter_cache[fighter_id]
        db = self._get_db()
        with db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM fighters WHERE id=?", (fighter_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Fighter id {fighter_id} not found in DB")
        fighter = dict(row)
        self._fighter_cache[fighter_id] = fighter
        return fighter

    def _get_fighters_in_class(self, weight_class: str) -> list[dict]:
        db = self._get_db()
        fighters = db.get_fighters_by_weight_class(weight_class)
        for f in fighters:
            self._fighter_cache[f["id"]] = f
        return fighters

    # ── Scoring ──────────────────────────────────────────────────────────

    def score_matchup(self, fighter_a_id: int, fighter_b_id: int) -> ScoredMatchup:
        """
        Score a single pairing by entertainment probability.

        Returns a ScoredMatchup with probability, star rating, top factors,
        style summary, and raw feature values for downstream use.
        """
        a = self._get_fighter(fighter_a_id)
        b = self._get_fighter(fighter_b_id)
        return self._score_pair(a, b)

    def _score_pair(self, a: dict, b: dict) -> ScoredMatchup:
        vec = _build_symmetric_vector(a, b, self.fe).reshape(1, -1)
        probs = _model_predict_proba(self.model, vec, self.scaler)
        prob = float(probs[0])
        stars = star_rating(prob)

        return ScoredMatchup(
            fighter_a=a.get("name", f"ID {a.get('id')}"),
            fighter_b=b.get("name", f"ID {b.get('id')}"),
            fighter_a_id=a.get("id", 0),
            fighter_b_id=b.get("id", 0),
            entertainment_probability=prob,
            entertainment_rating=stars,
            top_factors=_top_factors(a, b),
            style_summary=_style_summary(a, b),
            raw_features={"fighter_a": a, "fighter_b": b},
        )

    def _score_batch(self, pairs: list[tuple[dict, dict]]) -> list[ScoredMatchup]:
        """Vectorised scoring for many pairs at once."""
        if not pairs:
            return []

        vecs = np.stack([
            _build_symmetric_vector(a, b, self.fe) for a, b in pairs
        ])
        probs = _model_predict_proba(self.model, vecs, self.scaler)

        results = []
        for (a, b), prob in zip(pairs, probs):
            prob = float(prob)
            stars = star_rating(prob)
            results.append(ScoredMatchup(
                fighter_a=a.get("name", f"ID {a.get('id')}"),
                fighter_b=b.get("name", f"ID {b.get('id')}"),
                fighter_a_id=a.get("id", 0),
                fighter_b_id=b.get("id", 0),
                entertainment_probability=prob,
                entertainment_rating=stars,
                top_factors=_top_factors(a, b),
                style_summary=_style_summary(a, b),
                raw_features={"fighter_a": a, "fighter_b": b},
            ))
        return results

    # ── Ranking ──────────────────────────────────────────────────────────

    def rank_weight_class(
        self,
        weight_class: str,
        top_n: int = 10,
        min_fights: int = 3,
    ) -> list[ScoredMatchup]:
        """
        Score ALL unique pairings in a weight class and return the top N.

        For a division with k fighters this evaluates C(k,2) pairings.
        """
        fighters = self._get_fighters_in_class(weight_class)
        fighters = [
            f for f in fighters
            if (f.get("wins_total") or 0) + (f.get("losses_total") or 0) >= min_fights
        ]

        if len(fighters) < 2:
            logger.warning("Not enough fighters in %s to rank.", weight_class)
            return []

        pairs = list(combinations(fighters, 2))
        n_pairs = len(pairs)
        logger.info(
            "Ranking %s: %d fighters → %d unique pairings",
            weight_class, len(fighters), n_pairs,
        )

        t0 = time.perf_counter()
        results = self._score_batch(pairs)
        elapsed = time.perf_counter() - t0

        results.sort(key=lambda m: m.entertainment_probability, reverse=True)
        logger.info(
            "%s: scored %d pairings in %.2fs (%.0f pairings/sec)",
            weight_class, n_pairs, elapsed, n_pairs / max(elapsed, 1e-6),
        )
        return results[:top_n]

    # ── Card builder ─────────────────────────────────────────────────────

    def build_card(
        self,
        weight_classes: list[str],
        fights_per_class: int = 1,
        total_fights: int = 5,
    ) -> list[ScoredMatchup]:
        """
        Build a fight card across weight classes.

        Greedy algorithm: score all pairings across all requested divisions,
        then repeatedly pick the highest-probability matchup whose fighters
        haven't been used yet.

        Parameters
        ----------
        weight_classes     : divisions to draw from
        fights_per_class   : soft cap per division (0 = no cap)
        total_fights       : hard cap on total card size
        """
        all_results: list[ScoredMatchup] = []
        for wc in weight_classes:
            ranked = self.rank_weight_class(wc, top_n=50)
            all_results.extend(ranked)

        all_results.sort(key=lambda m: m.entertainment_probability, reverse=True)

        card: list[ScoredMatchup] = []
        used_ids: set[int] = set()
        class_counts: dict[str, int] = {}

        for matchup in all_results:
            if len(card) >= total_fights:
                break
            a_id = matchup.fighter_a_id
            b_id = matchup.fighter_b_id
            if a_id in used_ids or b_id in used_ids:
                continue

            # Determine weight class from raw data
            wc_a = matchup.raw_features.get("fighter_a", {}).get("weight_class", "")
            if fights_per_class > 0 and class_counts.get(wc_a, 0) >= fights_per_class:
                continue

            card.append(matchup)
            used_ids.add(a_id)
            used_ids.add(b_id)
            class_counts[wc_a] = class_counts.get(wc_a, 0) + 1

        logger.info("Card built: %d fights from %d weight classes", len(card), len(class_counts))
        return card

    # ── Narrative ────────────────────────────────────────────────────────

    def explain_matchup(self, fighter_a_id: int, fighter_b_id: int) -> str:
        """
        Generate a rich, human-readable narrative explaining WHY a matchup
        is (or isn't) predicted to be entertaining.

        Uses actual feature values to make each explanation unique.
        """
        a = self._get_fighter(fighter_a_id)
        b = self._get_fighter(fighter_b_id)
        scored = self._score_pair(a, b)
        return _generate_narrative(
            a, b,
            scored.entertainment_probability,
            scored.entertainment_rating,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rank all divisions
# ─────────────────────────────────────────────────────────────────────────────

def rank_all_divisions(
    matchmaker: MatchmakerV2,
    weight_classes: list[str],
    top_n_per_class: int = 5,
) -> dict[str, list[ScoredMatchup]]:
    """
    Run rank_weight_class for every division and return a master dict.
    """
    master: dict[str, list[ScoredMatchup]] = {}
    for wc in weight_classes:
        master[wc] = matchmaker.rank_weight_class(wc, top_n=top_n_per_class)
    return master


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_ranked_matchups(matchups: list[ScoredMatchup], title: str = "") -> None:
    """Print a clean table of ranked matchups."""
    if title:
        print(f"\n{'=' * 95}")
        print(f"  {title}")
        print(f"{'=' * 95}")

    header = (
        f"  {'Rank':<5} {'Fighter A':<20} {'Fighter B':<20} "
        f"{'Prob':>6} {'Stars':<7} {'Style Summary'}"
    )
    print(header)
    print("-" * 95)

    for i, m in enumerate(matchups, 1):
        print(
            f"  {i:<5} {m.fighter_a:<20} {m.fighter_b:<20} "
            f"{m.entertainment_probability:>5.1%} {star_string(m.entertainment_rating):<7} "
            f"{m.style_summary[:38]}"
        )
    print()


def print_card(card: list[ScoredMatchup], title: str = "SUGGESTED FIGHT CARD") -> None:
    """Print a fight card as a formatted table."""
    print(f"\n{'=' * 95}")
    print(f"  {title}")
    print(f"{'=' * 95}")

    for i, m in enumerate(card, 1):
        label = "MAIN EVENT" if i == 1 else f"Fight {i}"
        print(f"\n  [{label}]  {star_string(m.entertainment_rating)}  ({m.entertainment_probability:.0%})")
        print(f"    {m.fighter_a}  vs  {m.fighter_b}")
        print(f"    {m.style_summary}")
        for factor in m.top_factors:
            print(f"      - {factor}")
    print(f"\n{'=' * 95}")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fighter generator (for demo / testing)
# ─────────────────────────────────────────────────────────────────────────────

_FIRST_NAMES = [
    "Alex", "Brandon", "Carlos", "Diego", "Eduardo", "Felipe", "Gabriel",
    "Hector", "Ivan", "Jorge", "Kai", "Leon", "Marcus", "Nikolai",
    "Omar", "Paulo", "Rafael", "Santiago", "Timur", "Victor",
]
_LAST_NAMES = [
    "Silva", "Volkov", "Rodriguez", "Nakamura", "O'Brien", "Petrov",
    "Fernandez", "Kim", "Hassan", "Johansson", "Diaz", "Okafor",
    "Moreau", "Chen", "Alvarez", "Novak", "Torres", "Andersen",
    "Reyes", "Fischer",
]

_ARCHETYPES = {
    "ko_artist":   {"ko_rate": 0.65, "sub_rate": 0.05, "finish_rate": 0.70, "grapple_ratio": 0.15, "sig_strikes_pm": 6.0},
    "wrestler":    {"ko_rate": 0.10, "sub_rate": 0.25, "finish_rate": 0.35, "grapple_ratio": 0.70, "sig_strikes_pm": 2.5, "td_avg": 4.0},
    "submission":  {"ko_rate": 0.05, "sub_rate": 0.55, "finish_rate": 0.60, "grapple_ratio": 0.60, "sig_strikes_pm": 3.0, "sub_avg": 2.0},
    "volume":      {"ko_rate": 0.20, "sub_rate": 0.05, "finish_rate": 0.25, "grapple_ratio": 0.10, "sig_strikes_pm": 7.5},
    "balanced":    {"ko_rate": 0.30, "sub_rate": 0.15, "finish_rate": 0.45, "grapple_ratio": 0.35, "sig_strikes_pm": 4.5},
}


def generate_synthetic_fighters(
    n: int = 20,
    weight_class: str = "Lightweight",
    seed: int = 42,
) -> list[dict]:
    """Generate n synthetic fighters with varied styles for demo purposes."""
    rng = np.random.RandomState(seed)
    archetypes = list(_ARCHETYPES.keys())
    fighters = []

    for i in range(n):
        arch_name = archetypes[i % len(archetypes)]
        arch = _ARCHETYPES[arch_name].copy()
        noise = lambda: rng.uniform(-0.08, 0.08)

        wins = rng.randint(8, 25)
        losses = rng.randint(1, 8)
        ranking = i + 1 if i < 15 else None

        fighter = {
            "id": i + 1,
            "name": f"{_FIRST_NAMES[i % len(_FIRST_NAMES)]} {_LAST_NAMES[i % len(_LAST_NAMES)]}",
            "weight_class": weight_class,
            "ranking": ranking,
            "is_champion": 1 if i == 0 else 0,
            "wins_total": wins,
            "losses_total": losses,
            "height_cm": rng.uniform(170, 190),
            "reach_cm": rng.uniform(170, 200),
            "sig_strikes_pm": max(1.0, arch.get("sig_strikes_pm", 4.0) + rng.uniform(-1, 1)),
            "sig_strike_acc": np.clip(rng.uniform(0.38, 0.55), 0, 1),
            "sig_strike_def": np.clip(rng.uniform(0.45, 0.65), 0, 1),
            "sig_strikes_abs_pm": max(1.0, rng.uniform(2.0, 5.0)),
            "td_avg": max(0.0, arch.get("td_avg", 1.5) + rng.uniform(-0.5, 0.5)),
            "td_acc": np.clip(rng.uniform(0.30, 0.55), 0, 1),
            "td_def": np.clip(rng.uniform(0.50, 0.80), 0, 1),
            "sub_avg": max(0.0, arch.get("sub_avg", 0.5) + rng.uniform(-0.3, 0.3)),
            "ko_rate": np.clip(arch["ko_rate"] + noise(), 0, 1),
            "sub_rate": np.clip(arch["sub_rate"] + noise(), 0, 1),
            "dec_rate": 0.0,
            "finish_rate": np.clip(arch["finish_rate"] + noise(), 0, 1),
            "grapple_ratio": np.clip(arch["grapple_ratio"] + noise(), 0, 1),
            "ctrl_time_avg": rng.uniform(30, 180),
        }
        fighter["dec_rate"] = max(0.0, 1.0 - fighter["ko_rate"] - fighter["sub_rate"])
        fighters.append(fighter)

    return fighters


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import importlib.util
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    np.random.seed(42)

    # Load baselines module directly (avoids torch-dependent __init__.py)
    baselines_path = Path(__file__).parent / "baselines.py"
    spec = importlib.util.spec_from_file_location("baselines", baselines_path)
    baselines = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baselines)

    fe_path = Path(__file__).parent / "feature_engineering.py"
    spec_fe = importlib.util.spec_from_file_location("feature_engineering", fe_path)
    fe = importlib.util.module_from_spec(spec_fe)
    spec_fe.loader.exec_module(fe)

    # ── Train a quick XGBoost model ─────────────────────────────────────
    try:
        from models.data_loader import load_real_data
        data = load_real_data()
        print("  [Using REAL data from DB]")
    except (FileNotFoundError, ValueError, ImportError):
        data = baselines.load_placeholder_data()
        print("  [Using PLACEHOLDER data — DB not available]")

    print("Training XGBoost for demo...")
    bc = baselines.BaselineComparison()
    bc.load_data(data)
    bc.train_all()
    _, xgb_model = bc.get_best_model()

    # ── Generate synthetic fighters ──────────────────────────────────────
    fighters = generate_synthetic_fighters(n=20, weight_class="Lightweight")
    cache = {f["id"]: f for f in fighters}

    # ── Create matchmaker (no DB needed — we pass a fighter cache) ───────
    mm = MatchmakerV2(
        model=xgb_model,
        scaler=bc.scaler,
        feature_engineering_module=fe,
        fighter_cache=cache,
    )

    # ── Score all pairings ───────────────────────────────────────────────
    pairs = list(combinations(fighters, 2))
    print(f"\nScoring {len(pairs)} pairings across 20 synthetic fighters...")

    t0 = time.perf_counter()
    results = mm._score_batch(pairs)
    elapsed = time.perf_counter() - t0

    results.sort(key=lambda m: m.entertainment_probability, reverse=True)
    print(f"Scored in {elapsed:.3f}s ({len(pairs) / max(elapsed, 1e-6):.0f} pairings/sec)")

    # ── Show top 10 ──────────────────────────────────────────────────────
    print_ranked_matchups(results[:10], title="TOP 10 PREDICTED MATCHUPS — Lightweight Division")

    # ── Build a 5-fight card ─────────────────────────────────────────────
    all_results_sorted = sorted(results, key=lambda m: m.entertainment_probability, reverse=True)
    card: list[ScoredMatchup] = []
    used: set[int] = set()
    for m in all_results_sorted:
        if len(card) >= 5:
            break
        if m.fighter_a_id not in used and m.fighter_b_id not in used:
            card.append(m)
            used.add(m.fighter_a_id)
            used.add(m.fighter_b_id)
    print_card(card)

    # ── Explain the top matchup ──────────────────────────────────────────
    top = results[0]
    narrative = _generate_narrative(
        top.raw_features["fighter_a"],
        top.raw_features["fighter_b"],
        top.entertainment_probability,
        top.entertainment_rating,
    )
    print(f"\n  MATCHUP DEEP-DIVE: {top.fighter_a} vs {top.fighter_b}")
    print(f"  {'-' * 70}")
    print(f"  {narrative}")
    print()
