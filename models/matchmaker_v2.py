"""
models/matchmaker_v2.py
The product: an entertainment-optimised matchmaking engine.

**Default backend** is ``xgb_tuned_12feat.pkl`` blended with ``hgb_12feat_blend.pkl``:
``P = w * P_xgb + (1-w) * P_hgb`` on symmetric raw 12-D inputs (default ``w=0.99``).
Shallow XGB alone maps many pairs to identical probabilities; the HGB mix restores
ranking granularity and improves held-out AUC. Set ``MATCHMAKER_XGB_BLEND=1`` for
pure XGB, or run ``python -m models.fit_hgb_blend`` to build the HGB artifact.

Optional ``backend=\"nn\"`` loads ``FightBonusNN`` + ``scaler_12feat.pkl`` (legacy path).

Inference pipeline per pairing:
    fighter_a stats + fighter_b stats
        → build_full_matchup_vector()      [115-dim]
        → subset_full_feature_vector()      [12-dim raw]
        → (XGB+HGB blend) predict_proba(raw)  OR  (NN) scaler → sigmoid(logit)

Both orderings (A,B) and (B,A) are scored and the probabilities averaged so
the result is symmetric.
"""
from __future__ import annotations

import itertools
import logging
import os
from collections import defaultdict
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from models.feature_engineering import (
    ALL_FEATURE_NAMES,
    build_full_matchup_vector,
    extract_matchup_features,
    make_hypothetical_fight_context,
    subset_full_feature_vector,
)
from models.nn_binary import FightBonusNN
from models.pipeline_config import SELECTED_FEATURES
from models.rolling_features import _load_stats_dataframe, rolling_vector_asof

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_NN = Path(__file__).parent / "checkpoints" / "nn_12feat.pt"
DEFAULT_CHECKPOINT_XGB = Path(__file__).parent / "checkpoints" / "xgb_tuned_12feat.pkl"
DEFAULT_HGB_BLEND = Path(__file__).parent / "checkpoints" / "hgb_12feat_blend.pkl"
DEFAULT_SCALER = Path(__file__).parent / "checkpoints" / "scaler_12feat.pkl"
DEFAULT_DB = Path(__file__).parent.parent / "data" / "ufc_matchmaker.db"
# Weight on XGB in ``w * xgb + (1-w) * hgb``. Override with env MATCHMAKER_XGB_BLEND.
DEFAULT_XGB_BLEND_WEIGHT = 0.99

# UFC divisions in canonical form. Used both for filtering and for normalising
# ``fights.weight_class`` strings ("Lightweight Bout", "UFC Lightweight Title
# Bout", "Women's Strawweight Bout", …) into one of these canonical names.
CANONICAL_DIVISIONS = (
    "Strawweight",
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
)

# Match longest division names first so "Light Heavyweight" beats "Lightweight".
_DIVISION_PATTERNS = sorted(CANONICAL_DIVISIONS, key=len, reverse=True)


def _canonical_division(raw: str | None) -> str | None:
    """Map ``fights.weight_class`` strings to one of CANONICAL_DIVISIONS, or None."""
    if not raw:
        return None
    s = raw.strip()
    # Strip leading "UFC " (title bouts).
    s = re.sub(r"^UFC\s+", "", s)
    # Strip trailing " Bout" / " Title Bout".
    s = re.sub(r"\s+(Title\s+)?Bout$", "", s, flags=re.IGNORECASE).strip()
    for div in _DIVISION_PATTERNS:
        if s.lower().startswith(div.lower()):
            return div
    return None


def _matchup_sort_key(r: dict[str, Any]) -> tuple[float, float, int, int]:
    """
    Sort best-first. With the XGB+HGB blend, ``probability`` is usually unique
    enough; ``tiebreak`` remains a last resort for exact float ties.
    """
    a_id = int(r["fighter_a_id"])
    b_id = int(r["fighter_b_id"])
    lo, hi = (a_id, b_id) if a_id <= b_id else (b_id, a_id)
    return (
        -float(r["probability"]),
        -float(r.get("tiebreak", 0.0)),
        -lo,
        -hi,
    )


class MatchmakerV2:
    """
    The matchmaker. Self-contained: loads model + scaler + DB on init.

    Usage:
        mm = MatchmakerV2()                      # default: XGB pipeline
        mm.rank_weight_class("Lightweight", top_n=10)

        mm_nn = MatchmakerV2(backend="nn", ...)  # legacy NN + global scaler
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
        backend: str = "xgb",
        checkpoint_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
    ) -> None:
        self.backend = backend.strip().lower()
        if self.backend not in ("xgb", "nn"):
            raise ValueError("backend must be 'xgb' or 'nn'")

        self.xgb_pipeline = None
        self.hgb_blend_pipeline = None
        self._xgb_blend_weight = 1.0
        self.model = None
        self.scaler = None
        self.input_dim = 0

        # ── Model ─────────────────────────────────────────────────────────
        if self.backend == "xgb":
            ckpt_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_XGB)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"Missing XGB pipeline: {ckpt_path}\n"
                    "Train with: python -m models.xgb_tuning",
                )
            self.xgb_pipeline = joblib.load(ckpt_path)
            self.input_dim = int(
                self.xgb_pipeline.named_steps["xgb"].n_features_in_,
            )

            blend_env = os.environ.get("MATCHMAKER_XGB_BLEND", "").strip()
            self._xgb_blend_weight = (
                float(blend_env) if blend_env else float(DEFAULT_XGB_BLEND_WEIGHT)
            )
            if not 0.0 <= self._xgb_blend_weight <= 1.0:
                raise ValueError("MATCHMAKER_XGB_BLEND must be in [0, 1]")
            if self._xgb_blend_weight < 1.0:
                if DEFAULT_HGB_BLEND.is_file():
                    self.hgb_blend_pipeline = joblib.load(DEFAULT_HGB_BLEND)
                else:
                    logger.warning(
                        "MATCHMAKER_XGB_BLEND=%s but %s missing — run "
                        "`python -m models.fit_hgb_blend`. Using pure XGB.",
                        self._xgb_blend_weight,
                        DEFAULT_HGB_BLEND,
                    )
                    self._xgb_blend_weight = 1.0
        else:
            ckpt_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_NN)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint: {ckpt_path}\n"
                    "Generate it with: python -m models.nn_binary "
                    "(or call run_twelve_feature_comparison() programmatically)."
                )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            self.model = FightBonusNN(
                input_dim=cfg["input_dim"],
                hidden_dims=tuple(cfg["hidden_dims"]),
                dropout=cfg["dropout"],
            )
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self.input_dim = int(cfg["input_dim"])

            sc_path = Path(scaler_path or DEFAULT_SCALER)
            if not sc_path.is_file():
                raise FileNotFoundError(
                    f"Missing scaler: {sc_path}\n"
                    "Generate it by retraining: python -m models.nn_binary"
                )
            self.scaler = joblib.load(sc_path)

        # ── DB ────────────────────────────────────────────────────────────
        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row

        self._selected_features = list(SELECTED_FEATURES) if SELECTED_FEATURES else None
        if self._selected_features and len(self._selected_features) != self.input_dim:
            raise ValueError(
                f"SELECTED_FEATURES has {len(self._selected_features)} names but "
                f"model expects {self.input_dim}-dim input."
            )

        self._preload()

    # ── Preload ──────────────────────────────────────────────────────────

    def _preload(self) -> None:
        """Load all fighters + cache rolling vectors and division mapping."""
        self.fighters: dict[int, dict] = {}
        for row in self.db.execute("SELECT * FROM fighters"):
            self.fighters[int(row["id"])] = dict(row)

        # Rolling vectors: same *_rows_from_slice* path as training (not per-fighter SQL),
        # one dataframe load for all fighters.
        self.rolling_cache: dict[int, np.ndarray] = {}
        df = _load_stats_dataframe(self.db_path)
        grouped = (
            {k: v for k, v in df.groupby("fighter_id")} if not df.empty else {}
        )
        asof_ts = pd.Timestamp.now(tz="UTC")
        for fid, career in self.fighters.items():
            try:
                self.rolling_cache[int(fid)] = rolling_vector_asof(
                    int(fid),
                    grouped,
                    asof_ts,
                    dict(career),
                )
            except Exception as exc:
                logger.debug("rolling vec failed for fighter %s: %s", fid, exc)

        # Derive each fighter's primary division and last-event-date from fights.
        # ``fighters.weight_class`` is unpopulated in the current DB, so we infer
        # the division from the bouts they actually fought in.
        self._fighter_division: dict[int, str] = {}
        self._fighter_last_event: dict[int, str] = {}
        rows = self.db.execute(
            """
            SELECT f.fighter1_id AS fid, f.weight_class AS wc, e.date AS d
              FROM fights f JOIN events e ON e.id = f.event_id
            UNION ALL
            SELECT f.fighter2_id AS fid, f.weight_class AS wc, e.date AS d
              FROM fights f JOIN events e ON e.id = f.event_id
            """,
        ).fetchall()
        # Tally per (fighter, canonical division) and track most recent event.
        counts: dict[int, dict[str, int]] = {}
        for r in rows:
            fid = r["fid"]
            div = _canonical_division(r["wc"])
            d = r["d"] or ""
            if fid is None or div is None:
                continue
            counts.setdefault(int(fid), {})
            counts[int(fid)][div] = counts[int(fid)].get(div, 0) + 1
            prev = self._fighter_last_event.get(int(fid), "")
            if d and d > prev:
                self._fighter_last_event[int(fid)] = d
        for fid, divcounts in counts.items():
            self._fighter_division[fid] = max(divcounts.items(), key=lambda kv: kv[1])[0]

        n_with_roll = sum(1 for v in self.rolling_cache.values() if v is not None)
        n_with_div = len(self._fighter_division)
        print(
            f"Loaded {len(self.fighters)} fighters "
            f"({n_with_roll} with rolling stats, {n_with_div} mapped to a division)"
        )

    # ── Vector building ──────────────────────────────────────────────────

    def _raw_subvector(
        self,
        fa_id: int,
        fb_id: int,
        is_five_rounder: bool = False,
    ) -> np.ndarray:
        """12-D raw feature vector (same scale as training before global StandardScaler)."""
        fa = dict(self.fighters[fa_id])
        fb = dict(self.fighters[fb_id])
        fa["_rolling_vec"] = self.rolling_cache.get(fa_id)
        fb["_rolling_vec"] = self.rolling_cache.get(fb_id)

        fa["_fight_odds"] = None
        fb["_fight_odds"] = None

        wc = self._fighter_division.get(fa_id) or self._fighter_division.get(fb_id) or ""
        ctx = make_hypothetical_fight_context(
            is_title_fight=is_five_rounder,
            is_main_event=is_five_rounder,
            scheduled_rounds=5 if is_five_rounder else 3,
            weight_class=wc,
        )
        fa["_fight_context"] = ctx
        fb["_fight_context"] = ctx

        vec_115 = build_full_matchup_vector(fa, fb)
        vec_n = subset_full_feature_vector(
            vec_115, self._selected_features, list(ALL_FEATURE_NAMES),
        )
        return np.asarray(vec_n, dtype=np.float64).ravel()

    def _tiebreak_score(
        self,
        fa_id: int,
        fb_id: int,
        is_five_rounder: bool = False,
    ) -> float:
        """
        Symmetric secondary score when the classifier assigns the same probability
        to many pairs (e.g. XGB depth-2 leaves). Uses mean L1 norm of the raw
        model-input vectors for both corner orderings.
        """
        raw_ab = self._raw_subvector(fa_id, fb_id, is_five_rounder)
        raw_ba = self._raw_subvector(fb_id, fa_id, is_five_rounder)
        return (
            float(np.linalg.norm(raw_ab, ord=1))
            + float(np.linalg.norm(raw_ba, ord=1))
        ) / 2.0

    def _build_vector(
        self,
        fa_id: int,
        fb_id: int,
        is_five_rounder: bool = False,
    ) -> np.ndarray:
        """Scaled (1, input_dim) for NN backend; raises if backend is XGB."""
        if self.backend != "nn" or self.scaler is None:
            raise RuntimeError(
                "_build_vector (scaled) is only valid for backend='nn'. "
                "Use _raw_subvector() for raw 12-D features.",
            )
        raw = self._raw_subvector(fa_id, fb_id, is_five_rounder)
        return self.scaler.transform(raw.reshape(1, -1).astype(np.float32))

    def _predict_proba_nn(self, vec_scaled: np.ndarray) -> float:
        """NN forward + sigmoid (vec_scaled shape (1, n))."""
        if self.model is None:
            raise RuntimeError("NN not loaded")
        with torch.no_grad():
            logits = self.model(torch.from_numpy(vec_scaled.astype(np.float32)))
            return float(torch.sigmoid(logits).item())

    # ── Scoring ──────────────────────────────────────────────────────────

    def score_matchup(
        self,
        fa_id: int,
        fb_id: int,
        is_five_rounder: bool = False,
    ) -> dict:
        """Score one matchup. Averages both orderings for symmetry."""
        raw_ab = self._raw_subvector(fa_id, fb_id, is_five_rounder)
        raw_ba = self._raw_subvector(fb_id, fa_id, is_five_rounder)
        if self.backend == "xgb" and self.xgb_pipeline is not None:
            prob_xgb_ab = float(
                self.xgb_pipeline.predict_proba(raw_ab.reshape(1, -1))[0, 1],
            )
            prob_xgb_ba = float(
                self.xgb_pipeline.predict_proba(raw_ba.reshape(1, -1))[0, 1],
            )
            prob_xgb = (prob_xgb_ab + prob_xgb_ba) / 2.0
            if self.hgb_blend_pipeline is not None and self._xgb_blend_weight < 1.0:
                w = float(self._xgb_blend_weight)
                prob_hgb_ab = float(
                    self.hgb_blend_pipeline.predict_proba(
                        raw_ab.reshape(1, -1),
                    )[0, 1],
                )
                prob_hgb_ba = float(
                    self.hgb_blend_pipeline.predict_proba(
                        raw_ba.reshape(1, -1),
                    )[0, 1],
                )
                prob_hgb = (prob_hgb_ab + prob_hgb_ba) / 2.0
                prob = w * prob_xgb + (1.0 - w) * prob_hgb
            else:
                prob = prob_xgb
        else:
            prob_ab = self._predict_proba_nn(
                self.scaler.transform(raw_ab.reshape(1, -1).astype(np.float32)),
            )
            prob_ba = self._predict_proba_nn(
                self.scaler.transform(raw_ba.reshape(1, -1).astype(np.float32)),
            )
            prob = (prob_ab + prob_ba) / 2.0
        tiebreak = self._tiebreak_score(fa_id, fb_id, is_five_rounder)

        # Symmetric display: alphabetical by name so fighter_a isn't always the
        # first row in combinations() (sorted-by-name list → same corner every time).
        fa = self.fighters[fa_id]
        fb = self.fighters[fb_id]
        na = (fa.get("name") or "").strip().lower()
        nb = (fb.get("name") or "").strip().lower()
        if nb < na:
            fa_id, fb_id = fb_id, fa_id
            fa, fb = fb, fa

        stars = (
            5 if prob >= 0.65
            else 4 if prob >= 0.50
            else 3 if prob >= 0.35
            else 2 if prob >= 0.20
            else 1
        )
        return {
            "fighter_a": fa["name"],
            "fighter_b": fb["name"],
            "fighter_a_id": fa_id,
            "fighter_b_id": fb_id,
            "probability": float(prob),
            "tiebreak": float(tiebreak),
            "stars": stars,
            "reasons": self._explain(fa_id, fb_id),
        }

    def _explain(self, fa_id: int, fb_id: int) -> list[str]:
        """Human-readable reasons drawn from raw feature values."""
        fa = self.fighters[fa_id]
        fb = self.fighters[fb_id]
        ra = self.rolling_cache.get(fa_id)
        rb = self.rolling_cache.get(fb_id)
        reasons: list[str] = []

        # Style clash from cross-features (24-dim vector; index 0 is style_clash_score).
        try:
            cross = extract_matchup_features(fa, fb)
            if cross is not None and len(cross) > 0 and float(cross[0]) > 0.6:
                reasons.append("Contrasting styles create high-action potential")
        except Exception:
            pass

        # Recent knockdowns. Rolling layout (15-dim, see ROLLING_FIGHTER_FEATURE_NAMES):
        #   0 win_pct, 1 momentum, 2 form, 3 striking_diff_norm, 4 grappling_diff_norm,
        #   5 recent_sig_strikes_pm, 6 recent_strike_acc, 7 recent_takedowns_pm,
        #   8 recent_ctrl_norm, 9 recent_knockdowns_norm, 10 recent_subs_norm,
        #   11 strike_trend_norm, 12 damage_trend_norm, 13 performance_consistency,
        #   14 finish_streak.
        if ra is not None and rb is not None and len(ra) >= 15 and len(rb) >= 15:
            if ra[9] > 0.4:
                reasons.append(f"{fa['name']} has knockout power on display recently")
            if rb[9] > 0.4:
                reasons.append(f"{fb['name']} has knockout power on display recently")
            if ra[13] < 0.5 or rb[13] < 0.5:
                reasons.append("Unpredictable fighters — anything can happen")
            combined_output = float(ra[5] + rb[5]) * 12.0  # de-normalise (0..12 sspm)
            if combined_output > 7.0:
                reasons.append("Both fighters produce high-volume action")
            if ra[12] > 0.5 or rb[12] > 0.5:
                reasons.append("Recent fights suggest willingness to trade")

        if not reasons:
            reasons.append("Competitive matchup with balanced skill sets")
        return reasons[:3]

    # ── Active fighter lookup ────────────────────────────────────────────

    def _get_active_fighters(
        self,
        weight_class: str,
        within_years: int = 2,
    ) -> list[dict]:
        """
        Fighters whose primary division matches ``weight_class`` and who fought at
        least once within the last ``within_years``.

        Division is derived from ``fights.weight_class`` because ``fighters.weight_class``
        is unpopulated in this DB.
        """
        cutoff_row = self.db.execute(
            "SELECT date('now', ?)", (f"-{within_years} years",),
        ).fetchone()
        cutoff = cutoff_row[0] if cutoff_row else ""
        target = weight_class.strip()
        out: list[dict] = []
        for fid, div in self._fighter_division.items():
            if div != target:
                continue
            if cutoff and self._fighter_last_event.get(fid, "") < cutoff:
                continue
            out.append(self.fighters[fid])
        out.sort(key=lambda f: (f.get("name") or ""))
        return out

    # ── Ranking ──────────────────────────────────────────────────────────

    def rank_weight_class(
        self,
        weight_class: str,
        top_n: int = 10,
        is_five_rounder: bool = False,
        *,
        diverse: bool = True,
    ) -> list[dict]:
        """
        Score every unique pairing in a division and return the top N.

        If ``diverse`` is True (default), greedily prefers matchups that do not
        reuse a fighter already chosen, so the list is not ten variants of the
        same person vs everyone else (common when one name sorts first and scores high).
        """
        fighters = self._get_active_fighters(weight_class)
        pairs = list(itertools.combinations(fighters, 2))

        print(f"\nEvaluating {len(pairs)} possible matchups in {weight_class}...")
        results: list[dict] = []
        for fa, fb in pairs:
            try:
                results.append(self.score_matchup(int(fa["id"]), int(fb["id"]), is_five_rounder))
            except Exception as exc:
                logger.debug("scoring %s vs %s failed: %s", fa.get("name"), fb.get("name"), exc)

        results.sort(key=_matchup_sort_key)

        if diverse:
            picked: list[dict] = []
            used_ids: set[int] = set()
            for r in results:
                if r["fighter_a_id"] in used_ids or r["fighter_b_id"] in used_ids:
                    continue
                picked.append(r)
                used_ids.add(r["fighter_a_id"])
                used_ids.add(r["fighter_b_id"])
                if len(picked) >= top_n:
                    break
            if len(picked) < top_n:
                for r in results:
                    if r in picked:
                        continue
                    picked.append(r)
                    if len(picked) >= top_n:
                        break
            results_out = picked[:top_n]
        else:
            results_out = results[:top_n]

        print(f"\n{'=' * 75}")
        print(f"  {weight_class.upper()} — Top {min(top_n, len(results_out))} Most Entertaining Matchups")
        print(f"{'=' * 75}")
        for i, r in enumerate(results_out, 1):
            stars_str = "★" * r["stars"] + "☆" * (5 - r["stars"])
            print(f"\n  #{i}  {r['fighter_a']} vs {r['fighter_b']}")
            print(f"       {stars_str}  ({r['probability']:.2%} entertainment probability)")
            for reason in r["reasons"]:
                print(f"       • {reason}")
        print(f"\n{'=' * 75}")
        print(f"  Evaluated {len(pairs)} matchups from {len(fighters)} active fighters")
        print(f"{'=' * 75}")

        return results_out

    # ── Card builder ─────────────────────────────────────────────────────

    def build_card(
        self,
        weight_classes: Optional[list[str]] = None,
        total_fights: int = 5,
        *,
        max_per_weight_class: Optional[int] = 1,
    ) -> list[dict]:
        """
        Best fights across divisions. Each fighter appears at most once on the card
        (greedy pass in descending score order).

        ``max_per_weight_class`` limits how many bouts from the same division may
        appear (default 1 so the card spreads across weight classes). Use ``None``
        for no division cap (pure global top scores). If the cap prevents filling
        ``total_fights``, a second pass relaxes the division limit while still
        enforcing unique fighters.
        """
        if weight_classes is None:
            weight_classes = [
                "Lightweight", "Welterweight", "Middleweight",
                "Featherweight", "Bantamweight", "Light Heavyweight",
                "Flyweight", "Heavyweight",
            ]

        cap_note = (
            f" (max {max_per_weight_class} fight(s) per division)"
            if max_per_weight_class is not None
            else " (no per-division cap)"
        )
        print(f"\nBuilding dream card across {len(weight_classes)} divisions{cap_note}...")

        all_matchups: list[dict] = []
        for wc in weight_classes:
            fighters = self._get_active_fighters(wc)
            for fa, fb in itertools.combinations(fighters, 2):
                try:
                    r = self.score_matchup(int(fa["id"]), int(fb["id"]))
                    r["weight_class"] = wc
                    all_matchups.append(r)
                except Exception:
                    continue

        all_matchups.sort(key=_matchup_sort_key)

        def pair_key(m: dict) -> frozenset[int]:
            return frozenset((m["fighter_a_id"], m["fighter_b_id"]))

        card: list[dict] = []
        used: set[int] = set()
        chosen: set[frozenset[int]] = set()
        wc_counts: defaultdict[str, int] = defaultdict(int)

        def consider(m: dict, *, enforce_div_cap: bool) -> bool:
            k = pair_key(m)
            if k in chosen:
                return False
            if m["fighter_a_id"] in used or m["fighter_b_id"] in used:
                return False
            wc = m["weight_class"]
            if enforce_div_cap and max_per_weight_class is not None:
                if wc_counts[wc] >= max_per_weight_class:
                    return False
            card.append(m)
            chosen.add(k)
            used.add(m["fighter_a_id"])
            used.add(m["fighter_b_id"])
            wc_counts[wc] += 1
            return True

        for m in all_matchups:
            if len(card) >= total_fights:
                break
            consider(m, enforce_div_cap=True)

        if max_per_weight_class is not None and len(card) < total_fights:
            for m in all_matchups:
                if len(card) >= total_fights:
                    break
                consider(m, enforce_div_cap=False)

        print(f"\n{'=' * 75}")
        print(f"  AI-GENERATED DREAM CARD — {len(card)} Most Entertaining Fights")
        print(f"{'=' * 75}")
        for i, r in enumerate(card, 1):
            stars_str = "★" * r["stars"] + "☆" * (5 - r["stars"])
            print(f"\n  Fight {i}: {r['fighter_a']} vs {r['fighter_b']}")
            print(f"           {r['weight_class']} | {stars_str} | {r['probability']:.2%}")
            print(f"           {r['reasons'][0]}")
        print(f"\n{'=' * 75}")
        return card


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    mm = MatchmakerV2()
    mm.rank_weight_class("Lightweight", top_n=10)
    mm.build_card(total_fights=5)
