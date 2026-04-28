"""
models/matchmaker.py
═══════════════════════════════════════════════════════════════════════════════
Matchmaking Engine

Given a pool of fighters (usually a weight class), this module generates every
valid fighter pairing, scores each one with the trained NN, and returns a ranked
list of "best matchups" from a business / entertainment perspective.

A matchup is scored along two axes:
  1. NN score      — the neural network's predicted fight quality (0-100)
  2. Business score — a hand-crafted signal that weights rankings, name value,
                      and title implications on top of raw fight quality.

The final displayed score is a weighted blend of both.

Why a separate business score?
  The NN is trained purely on fight *quality* (action, finishes, balance). But
  the UFC as a business also cares about: is at least one fighter ranked?  Is
  there a title on the line?  Are both fighters on hot streaks?  These signals
  are outside the raw in-cage data but matter enormously for PPV buys and card
  placement.
═══════════════════════════════════════════════════════════════════════════════
"""
import logging
import pickle
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from data.db import Database
from models.feature_engineering import build_full_matchup_vector
from models.fight_quality_nn import FightQualityNN
from models.training import load_model, load_scaler
from config import NN, WEIGHT_CLASSES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes for clean output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FighterProfile:
    """
    Lightweight representation of a fighter used by the matchmaker.
    Populated from DB fighter rows.
    """
    id:            int
    name:          str
    weight_class:  str
    ranking:       Optional[int]      # None = unranked
    is_champion:   bool = False
    wins_total:    int = 0
    losses_total:  int = 0
    finish_rate:   float = 0.5
    ko_rate:       float = 0.3
    sig_strikes_pm: float = 3.0
    # Raw dict kept for feature extraction
    raw:           dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_db_row(cls, row: dict) -> "FighterProfile":
        return cls(
            id=row["id"],
            name=row["name"],
            weight_class=row.get("weight_class", ""),
            ranking=row.get("ranking"),
            is_champion=bool(row.get("is_champion", 0)),
            wins_total=row.get("wins_total") or 0,
            losses_total=row.get("losses_total") or 0,
            finish_rate=row.get("finish_rate") or 0.5,
            ko_rate=row.get("ko_rate") or 0.3,
            sig_strikes_pm=row.get("sig_strikes_pm") or 3.0,
            raw=row,
        )

    @property
    def record(self) -> str:
        return f"{self.wins_total}-{self.losses_total}"

    @property
    def ranked_label(self) -> str:
        if self.is_champion:
            return "C"
        if self.ranking:
            return f"#{self.ranking}"
        return "NR"  # Not ranked


@dataclass
class MatchupResult:
    """
    A single predicted matchup with all scores and human-readable signals.
    """
    fighter_a:         FighterProfile
    fighter_b:         FighterProfile

    # Raw NN prediction (0-100)
    nn_score:          float

    # Business overlay score (0-100)
    business_score:    float

    # Final blended score (0-100) — used for ranking
    final_score:       float

    # Interpretable sub-scores (0-1 each, for display)
    style_clash:       float    # How stylistically different are they?
    finish_probability: float   # Estimated probability the fight gets finished
    competitive_balance: float  # How even is the matchup on paper?
    action_density:    float    # Predicted strikes-per-minute intensity
    title_implications: bool    # Does this fight move someone toward the belt?

    # Narrative descriptor (generated from features)
    narrative:         str = ""

    def star_rating(self, score: float, max_stars: int = 5) -> str:
        """Convert a 0-100 score to a star string like ★★★★☆."""
        filled = round((score / 100.0) * max_stars)
        return "★" * filled + "☆" * (max_stars - filled)

    def __lt__(self, other: "MatchupResult") -> bool:
        """Allows sorted() to work on MatchupResult lists."""
        return self.final_score < other.final_score


# ─────────────────────────────────────────────────────────────────────────────
# Matchmaking engine
# ─────────────────────────────────────────────────────────────────────────────

class Matchmaker:
    """
    Loads a trained FightQualityNN and scores all pairings of a fighter pool.

    Usage
    -----
    with Matchmaker(db) as mm:
        results = mm.predict_weight_class("Lightweight", top_n=20)
        mm.print_report(results)
    """

    # Blend weights: how much of the final score comes from the NN vs business signals
    NN_WEIGHT       = 0.70
    BUSINESS_WEIGHT = 0.30

    def __init__(
        self,
        db: Database,
        model_path: str = None,
        scaler_path: str = None,
    ):
        self.db           = db
        self.model_path   = model_path or NN["model_save_path"]
        self.scaler_path  = scaler_path or NN["scaler_save_path"]
        self._model: Optional[FightQualityNN]  = None
        self._scaler = None

    # ── Context manager so we load model once per session ────────────────────

    def __enter__(self):
        self._load_model()
        return self

    def __exit__(self, *_):
        self._model = None
        self._scaler = None

    def _load_model(self):
        """Load the trained NN and feature scaler from disk."""
        try:
            self._model  = load_model(self.model_path)
            self._scaler = load_scaler(self.scaler_path)
            logger.info("Matchmaker ready.")
        except FileNotFoundError:
            raise RuntimeError(
                "Trained model not found. Run training first:\n"
                "  python main.py train"
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_weight_class(
        self,
        weight_class: str,
        top_n: int = 20,
        min_fights: int = 3,
        ranked_only: bool = False,
    ) -> list[MatchupResult]:
        """
        Generate and rank all pairings within a weight class.

        Parameters
        ----------
        weight_class : e.g. "Lightweight"
        top_n        : Return only the top-N results
        min_fights   : Filter out fighters with fewer total fights (avoids
                       under-represented newcomers skewing results)
        ranked_only  : If True, only include fighters who are currently ranked

        Returns
        -------
        List of MatchupResult sorted by final_score descending.
        """
        fighters = self._load_fighters(weight_class, min_fights, ranked_only)

        if len(fighters) < 2:
            logger.warning("Not enough fighters in %s to make pairings.", weight_class)
            return []

        logger.info(
            "Scoring %d pairings for %s (%d fighters)...",
            len(fighters) * (len(fighters) - 1) // 2,
            weight_class,
            len(fighters),
        )

        results = []
        for fa, fb in combinations(fighters, 2):
            result = self._score_matchup(fa, fb)
            results.append(result)

        # Sort descending by final score
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_n]

    def predict_specific_matchup(
        self,
        fighter_a_name: str,
        fighter_b_name: str,
    ) -> Optional[MatchupResult]:
        """
        Score one specific named matchup.
        Useful for querying: "How good would Poirier vs. Chandler be?"
        """
        fa_row = self._get_fighter_by_name(fighter_a_name)
        fb_row = self._get_fighter_by_name(fighter_b_name)

        if fa_row is None:
            logger.error("Fighter not found: %s", fighter_a_name)
            return None
        if fb_row is None:
            logger.error("Fighter not found: %s", fighter_b_name)
            return None

        fa = FighterProfile.from_db_row(fa_row)
        fb = FighterProfile.from_db_row(fb_row)
        return self._score_matchup(fa, fb)

    def predict_card(
        self,
        weight_classes: list[str] = None,
        n_per_class: int = 3,
    ) -> dict[str, list[MatchupResult]]:
        """
        Build a suggested fight card by selecting the top pairings
        from each weight class, ensuring no fighter appears twice.

        Returns dict keyed by weight class.
        """
        if weight_classes is None:
            weight_classes = WEIGHT_CLASSES

        card: dict[str, list[MatchupResult]] = {}
        used_fighter_ids: set[int] = set()

        for wc in weight_classes:
            # Get more candidates than we need so we can skip already-used fighters
            candidates = self.predict_weight_class(wc, top_n=n_per_class * 5)
            selected = []
            for matchup in candidates:
                a_id = matchup.fighter_a.id
                b_id = matchup.fighter_b.id
                if a_id not in used_fighter_ids and b_id not in used_fighter_ids:
                    selected.append(matchup)
                    used_fighter_ids.add(a_id)
                    used_fighter_ids.add(b_id)
                if len(selected) >= n_per_class:
                    break
            if selected:
                card[wc] = selected

        return card

    # ── Core scoring logic ───────────────────────────────────────────────────

    def _score_matchup(
        self,
        fa: FighterProfile,
        fb: FighterProfile,
    ) -> MatchupResult:
        """
        Score a single matchup by:
          1. Building feature vector
          2. Scaling features
          3. Getting NN prediction
          4. Computing business overlay
          5. Blending and computing sub-scores for display
        """
        # ── Feature vector: 72-dim = fA(24) + fB(24) + cross(24), symmetric avg
        vec_ab = build_full_matchup_vector(fa.raw, fb.raw)
        vec_ba = build_full_matchup_vector(fb.raw, fa.raw)
        vec    = ((vec_ab + vec_ba) / 2.0).reshape(1, -1)

        # Scale using the fitted scaler
        vec_scaled = self._scaler.transform(vec).astype(np.float32)

        # ── NN score ─────────────────────────────────────────────────────────
        nn_score = float(self._model.predict_batch(vec_scaled)[0])
        nn_score = max(0.0, min(100.0, nn_score))

        # ── Business overlay ─────────────────────────────────────────────────
        business_score = self._compute_business_score(fa, fb)

        # ── Final blended score ───────────────────────────────────────────────
        final_score = (
            self.NN_WEIGHT       * nn_score
            + self.BUSINESS_WEIGHT * business_score
        )

        # ── Interpretable sub-scores ─────────────────────────────────────────
        style_clash          = self._style_clash(fa, fb)
        finish_probability   = self._finish_probability(fa, fb)
        competitive_balance  = self._competitive_balance(fa, fb)
        action_density       = self._action_density(fa, fb)
        title_implications   = self._has_title_implications(fa, fb)

        # ── Narrative ────────────────────────────────────────────────────────
        narrative = self._generate_narrative(
            fa, fb, style_clash, finish_probability,
            competitive_balance, action_density, title_implications,
        )

        return MatchupResult(
            fighter_a=fa,
            fighter_b=fb,
            nn_score=round(nn_score, 1),
            business_score=round(business_score, 1),
            final_score=round(final_score, 1),
            style_clash=style_clash,
            finish_probability=finish_probability,
            competitive_balance=competitive_balance,
            action_density=action_density,
            title_implications=title_implications,
            narrative=narrative,
        )

    # ── Business overlay components ──────────────────────────────────────────

    def _compute_business_score(self, fa: FighterProfile, fb: FighterProfile) -> float:
        """
        Business score (0-100) based on:
          - Ranking prominence (ranked fighters = more PPV interest)
          - Title implications (does the winner get a shot?)
          - Momentum (both fighters on winning streaks = more buzz)
          - Name value proxy (finish rate × record = excitement brand)

        These are deliberately separate from the NN because they're
        about *market value*, not purely in-cage quality.
        """
        score = 0.0

        # ── Ranking points (0-40) ────────────────────────────────────────────
        # Being ranked at all = 20 pts. Lower rank number = more pts.
        # Champion = full 40 pts.
        def rank_pts(f: FighterProfile) -> float:
            if f.is_champion:
                return 40.0
            if f.ranking is not None:
                # Rank 1 = 20 pts, rank 15 = ~7 pts
                return max(5.0, 20.0 - (f.ranking - 1) * (13.0 / 14.0))
            return 0.0  # Unranked = 0 business pts from rankings

        score += (rank_pts(fa) + rank_pts(fb)) / 2.0

        # ── Title implications (0-20) ────────────────────────────────────────
        # Both ranked in top 5 = clear title eliminator value
        both_top5 = (
            fa.ranking is not None and fa.ranking <= 5
            and fb.ranking is not None and fb.ranking <= 5
        )
        one_top5  = (
            (fa.ranking is not None and fa.ranking <= 5) or
            (fb.ranking is not None and fb.ranking <= 5)
        )
        if both_top5:
            score += 20.0
        elif one_top5:
            score += 10.0

        # ── Finish rate as "name value" proxy (0-25) ─────────────────────────
        # Fans know finishers. High finish rate = built-in excitement brand.
        avg_finish = (fa.finish_rate + fb.finish_rate) / 2.0
        score += avg_finish * 25.0

        # ── Record quality proxy (0-15) ──────────────────────────────────────
        # Both fighters having strong records suggests a legitimate high-level fight.
        def record_score(f: FighterProfile) -> float:
            total = max(f.wins_total + f.losses_total, 1)
            win_pct = f.wins_total / total
            # Bonus for having many fights (not a newcomer)
            experience_bonus = min(total / 20.0, 1.0) * 0.3
            return (win_pct + experience_bonus) / 1.3

        avg_record = (record_score(fa) + record_score(fb)) / 2.0
        score += avg_record * 15.0

        return min(score, 100.0)

    # ── Interpretable sub-scores (displayed in the output) ───────────────────

    def _style_clash(self, fa: FighterProfile, fb: FighterProfile) -> float:
        """
        How stylistically contrasting are these fighters?
        Striker vs. Grappler = high clash = more interesting.
        Two pure grapplers = low clash.
        Returns 0.0 – 1.0.
        """
        gr_a = fa.raw.get("grapple_ratio") or 0.3
        gr_b = fb.raw.get("grapple_ratio") or 0.3
        return min(abs(gr_a - gr_b) * 2.0, 1.0)

    def _finish_probability(self, fa: FighterProfile, fb: FighterProfile) -> float:
        """
        Rough estimate: if both fighters are high finishers,
        the fight is more likely to be finished.
        Formula: sqrt(finish_rate_A * finish_rate_B) rescaled.
        Returns 0.0 – 1.0.
        """
        return min((fa.finish_rate * fb.finish_rate) ** 0.5 * 1.5, 1.0)

    def _competitive_balance(self, fa: FighterProfile, fb: FighterProfile) -> float:
        """
        How even are the fighters on paper?
        Uses win% and ranking proximity.
        1.0 = perfectly matched, 0.0 = total mismatch.
        """
        total_a = max(fa.wins_total + fa.losses_total, 1)
        total_b = max(fb.wins_total + fb.losses_total, 1)
        wp_a = fa.wins_total / total_a
        wp_b = fb.wins_total / total_b
        wp_diff = abs(wp_a - wp_b)

        rank_diff = 0.0
        if fa.ranking is not None and fb.ranking is not None:
            rank_diff = min(abs(fa.ranking - fb.ranking) / 15.0, 1.0)

        # Balance = 1 when both metrics say "even match"
        balance = 1.0 - (wp_diff * 0.6 + rank_diff * 0.4)
        return max(0.0, min(1.0, balance))

    def _action_density(self, fa: FighterProfile, fb: FighterProfile) -> float:
        """
        Predicted pace of the fight based on combined striking output.
        Returns 0.0 – 1.0 (1.0 ≈ Holloway-level volume striking).
        """
        combined_spm = (fa.sig_strikes_pm or 3.0) + (fb.sig_strikes_pm or 3.0)
        # ~12 combined sig strikes/min is elite pace → normalise to that ceiling
        return min(combined_spm / 12.0, 1.0)

    def _has_title_implications(self, fa: FighterProfile, fb: FighterProfile) -> bool:
        """True if one fighter is champion OR both are ranked in the top 5."""
        if fa.is_champion or fb.is_champion:
            return True
        both_top5 = (
            fa.ranking is not None and fa.ranking <= 5
            and fb.ranking is not None and fb.ranking <= 5
        )
        return both_top5

    # ── Narrative generator ──────────────────────────────────────────────────

    def _generate_narrative(
        self,
        fa: FighterProfile,
        fb: FighterProfile,
        style_clash: float,
        finish_prob: float,
        balance: float,
        action: float,
        title: bool,
    ) -> str:
        """
        Generate a short, punchy fight description based on the sub-scores.
        These are rule-based so they work even without internet access.
        """
        parts = []

        # Style description
        gr_a = fa.raw.get("grapple_ratio") or 0.3
        gr_b = fb.raw.get("grapple_ratio") or 0.3
        if style_clash > 0.5:
            striker   = fa if gr_a < gr_b else fb
            grappler  = fb if gr_a < gr_b else fa
            parts.append(f"Pure striker {striker.name} vs elite grappler {grappler.name}")
        elif gr_a > 0.55 and gr_b > 0.55:
            parts.append("High-level grappling matchup — submission and ground-and-pound threats")
        elif gr_a < 0.3 and gr_b < 0.3:
            parts.append("Two high-output strikers on a collision course")
        else:
            parts.append("Well-rounded fighters with overlapping skill sets")

        # Pace
        if action > 0.75:
            parts.append("expect an extremely high-paced, action-packed contest")
        elif action > 0.5:
            parts.append("steady pace with consistent pressure throughout")

        # Finish signal
        if finish_prob > 0.65:
            parts.append("finish highly likely")
        elif finish_prob > 0.4:
            parts.append("finish possible but decision is realistic")
        else:
            parts.append("could go the distance")

        # Title / stakes
        if title:
            parts.append("with clear title implications")

        return ". ".join(p.capitalize() for p in parts) + "."

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _load_fighters(
        self,
        weight_class: str,
        min_fights: int,
        ranked_only: bool,
    ) -> list[FighterProfile]:
        """Load fighters from DB, apply filters, return FighterProfile list."""
        rows = self.db.get_fighters_by_weight_class(weight_class)

        filtered = []
        for row in rows:
            total_fights = (row.get("wins_total") or 0) + (row.get("losses_total") or 0)
            if total_fights < min_fights:
                continue
            if ranked_only and row.get("ranking") is None:
                continue
            filtered.append(FighterProfile.from_db_row(row))

        logger.debug("Loaded %d fighters for %s (min_fights=%d)", len(filtered), weight_class, min_fights)
        return filtered

    def _get_fighter_by_name(self, name: str) -> Optional[dict]:
        """Case-insensitive fighter name lookup."""
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM fighters WHERE LOWER(name) = LOWER(?)",
                (name,),
            ).fetchone()
            return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic-only matchmaker (no trained model needed)
# Used as fallback when no model is trained yet, or for quick exploration.
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicMatchmaker:
    """
    Scores matchups using only the hand-crafted business/style heuristics,
    without requiring a trained NN.  Useful for:
      - Exploring data before training
      - Sanity-checking the feature engineering
      - Running on a fresh DB with few fights
    """

    def __init__(self, db: Database):
        self.db = db

    def predict_specific_matchup(
        self,
        fighter_a_name: str,
        fighter_b_name: str,
    ) -> Optional[MatchupResult]:
        """Score one named pairing using the same heuristic as predict_weight_class."""
        fa_row = self._get_fighter_by_name(fighter_a_name)
        fb_row = self._get_fighter_by_name(fighter_b_name)
        if fa_row is None or fb_row is None:
            return None
        fa = FighterProfile.from_db_row(fa_row)
        fb = FighterProfile.from_db_row(fb_row)
        score = self._heuristic_score(fa, fb)
        gr_a = fa.raw.get("grapple_ratio") or 0.3
        gr_b = fb.raw.get("grapple_ratio") or 0.3
        return MatchupResult(
            fighter_a=fa,
            fighter_b=fb,
            nn_score=score,
            business_score=score,
            final_score=score,
            style_clash=min(abs(gr_a - gr_b) * 2.0, 1.0),
            finish_probability=min((fa.finish_rate * fb.finish_rate) ** 0.5 * 1.5, 1.0),
            competitive_balance=0.5,
            action_density=min(((fa.sig_strikes_pm or 3) + (fb.sig_strikes_pm or 3)) / 12, 1),
            title_implications=(
                fa.ranking is not None and fa.ranking <= 5
                and fb.ranking is not None and fb.ranking <= 5
            ),
            narrative=f"Heuristic matchup: {fa.name} vs {fb.name}",
        )

    def _get_fighter_by_name(self, name: str) -> Optional[dict]:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM fighters WHERE LOWER(name) = LOWER(?)",
                (name,),
            ).fetchone()
            return dict(row) if row else None

    def predict_weight_class(
        self,
        weight_class: str,
        top_n: int = 20,
        min_fights: int = 3,
    ) -> list[MatchupResult]:
        rows = self.db.get_fighters_by_weight_class(weight_class)
        fighters = [
            FighterProfile.from_db_row(r)
            for r in rows
            if (r.get("wins_total") or 0) + (r.get("losses_total") or 0) >= min_fights
        ]

        results = []
        for fa, fb in combinations(fighters, 2):
            score = self._heuristic_score(fa, fb)
            results.append(MatchupResult(
                fighter_a=fa,
                fighter_b=fb,
                nn_score=score,
                business_score=score,
                final_score=score,
                style_clash=abs((fa.raw.get("grapple_ratio") or 0.3) - (fb.raw.get("grapple_ratio") or 0.3)),
                finish_probability=(fa.finish_rate * fb.finish_rate) ** 0.5,
                competitive_balance=0.5,
                action_density=min(((fa.sig_strikes_pm or 3) + (fb.sig_strikes_pm or 3)) / 12, 1),
                title_implications=(fa.ranking is not None and fa.ranking <= 5
                                    and fb.ranking is not None and fb.ranking <= 5),
                narrative=f"Heuristic matchup: {fa.name} vs {fb.name}",
            ))

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_n]

    def _heuristic_score(self, fa: FighterProfile, fb: FighterProfile) -> float:
        """Simple weighted combination of the key fight quality signals."""
        from config import QUALITY_WEIGHTS as W

        # Action density (0-1)
        combined_spm = ((fa.sig_strikes_pm or 3) + (fb.sig_strikes_pm or 3)) / 12.0
        action = min(combined_spm, 1.0)

        # Finish probability
        finish = min((fa.finish_rate * fb.finish_rate) ** 0.5 * 1.5, 1.0)

        # Competitive balance
        total_a = max(fa.wins_total + fa.losses_total, 1)
        total_b = max(fb.wins_total + fb.losses_total, 1)
        wp_diff = abs(fa.wins_total / total_a - fb.wins_total / total_b)
        balance = 1.0 - wp_diff

        # Style clash
        gr_a = fa.raw.get("grapple_ratio") or 0.3
        gr_b = fb.raw.get("grapple_ratio") or 0.3
        clash = min(abs(gr_a - gr_b) * 2.0, 1.0)

        # Marketability
        def rank_val(f: FighterProfile) -> float:
            if f.is_champion: return 1.0
            if f.ranking: return max(0.0, (16 - f.ranking) / 15.0)
            return 0.0

        market = (rank_val(fa) + rank_val(fb)) / 2.0

        raw = (
            W["action_density"]    * action
            + W["finish_probability"]  * finish
            + W["competitive_balance"] * balance
            + W["style_clash"]         * clash
            + W["marketability"]       * market
        )
        return round(raw * 100.0, 1)
