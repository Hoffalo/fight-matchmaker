"""
models/backtesting.py
Lorenzo's L4 — backtesting analysis.

For each held-out UFC event, run a trained classifier on every fight, rank by
predicted FOTN/POTN probability, and check whether the model's top-k picks
include the fights that actually won bonuses.

The classifier is pluggable — anything with sklearn's `predict_proba` interface
works. For now we ship a LogisticRegression baseline so the framework runs
end-to-end before Raji's XGBoost/RF/NN land.
"""
import csv
import logging
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data.db import Database
from models.feature_engineering import build_matchup_vector
from config import BASE_DIR

logger = logging.getLogger(__name__)

OUTPUTS_DIR = BASE_DIR / "outputs"
BACKTEST_CSV = OUTPUTS_DIR / "backtest_results.csv"
BACKTEST_MD = OUTPUTS_DIR / "backtest_results.md"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_fighters(db: Database) -> dict[int, dict]:
    with db.connect() as conn:
        return {row["id"]: dict(row) for row in conn.execute("SELECT * FROM fighters")}


def _build_event_fights(db: Database, fighters: dict[int, dict]) -> list[dict]:
    """
    Return every usable fight as a dict with feature vector + metadata.
    """
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT f.id, f.fighter1_id, f.fighter2_id,
                      f.is_bonus_fight, e.id AS event_id, e.date, e.name AS event_name
               FROM fights f LEFT JOIN events e ON e.id = f.event_id
               WHERE f.fighter1_id IS NOT NULL AND f.fighter2_id IS NOT NULL"""
        ).fetchall()

    out: list[dict] = []
    for r in rows:
        f1 = fighters.get(r["fighter1_id"])
        f2 = fighters.get(r["fighter2_id"])
        if not (f1 and f2):
            continue
        # Use the (A,B) ordering for inference; symmetry is a training-time concern
        vec = build_matchup_vector(f1, f2)
        out.append({
            "fight_id":   r["id"],
            "event_id":   r["event_id"],
            "event_date": r["date"] or "",
            "event_name": r["event_name"],
            "fighter1":   f1.get("name", "?"),
            "fighter2":   f2.get("name", "?"),
            "label":      int(r["is_bonus_fight"] or 0),
            "X":          vec,
        })
    return out


# ── Baseline classifier (used until Raji's model lands) ─────────────────────

def train_baseline_classifier(
    train_rows: list[dict], random_state: int = 42
) -> tuple[LogisticRegression, StandardScaler]:
    X = np.stack([r["X"] for r in train_rows])
    y = np.array([r["label"] for r in train_rows], dtype=np.int64)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Augment symmetrically: every (A,B) row paired with its (B,A) twin.
    # train_rows already has only (A,B); reflect by swapping fighters at
    # build time — but we'd need access to fighter dicts again. Skip the
    # symmetric augmentation here since the L4 model is just for backtesting,
    # not the team's official baseline. This is a temporary classifier.

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    ).fit(Xs, y)
    return clf, scaler


def predict_proba(
    clf, scaler: StandardScaler, rows: list[dict]
) -> np.ndarray:
    X = np.stack([r["X"] for r in rows])
    Xs = scaler.transform(X)
    return clf.predict_proba(Xs)[:, 1]


# ── Backtesting loop ─────────────────────────────────────────────────────────

def backtest(
    db: Database,
    *,
    test_date_from: str = "2026-01-01",
    k: int = 3,
    random_state: int = 42,
) -> dict:
    """
    Time-respecting backtest: train on everything strictly before
    `test_date_from`, evaluate per-event on everything from that date onward.

    Returns summary dict and writes outputs/backtest_results.{csv,md}.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fighters = _load_fighters(db)
    all_rows = _build_event_fights(db, fighters)

    train_rows = [r for r in all_rows if r["event_date"] < test_date_from]
    test_rows  = [r for r in all_rows if r["event_date"] >= test_date_from]

    if not train_rows or not test_rows:
        raise RuntimeError(
            f"Empty train ({len(train_rows)}) or test ({len(test_rows)}) split "
            f"around {test_date_from}. Adjust test_date_from."
        )

    train_pos = sum(r["label"] for r in train_rows)
    test_pos = sum(r["label"] for r in test_rows)
    logger.info(
        "Train: %d fights (%d pos)  |  Test: %d fights (%d pos)",
        len(train_rows), train_pos, len(test_rows), test_pos,
    )

    clf, scaler = train_baseline_classifier(train_rows, random_state=random_state)

    # Group test rows by event
    by_event: dict[int, list[dict]] = {}
    for r in test_rows:
        by_event.setdefault(r["event_id"], []).append(r)

    per_event_records: list[dict] = []
    hits = 0
    total_actual_bonus_fights = 0

    for event_id, rows in sorted(
        by_event.items(), key=lambda kv: kv[1][0]["event_date"]
    ):
        probs = predict_proba(clf, scaler, rows)
        ranked = sorted(zip(rows, probs), key=lambda t: -t[1])
        top_k = ranked[:k]
        actual_bonus_fights = [r for r in rows if r["label"] == 1]

        # Hit if any top-k pick coincides with an actual bonus fight
        top_k_fight_ids = {r["fight_id"] for r, _ in top_k}
        actual_ids = {r["fight_id"] for r in actual_bonus_fights}
        local_hit = bool(top_k_fight_ids & actual_ids)
        if local_hit:
            hits += 1
        total_actual_bonus_fights += len(actual_bonus_fights)

        per_event_records.append({
            "event_date": rows[0]["event_date"],
            "event_name": rows[0]["event_name"],
            "num_fights": len(rows),
            "num_actual_bonus": len(actual_bonus_fights),
            "top_k_picks": [
                f"{r['fighter1']} vs {r['fighter2']} (p={p:.2f})" for r, p in top_k
            ],
            "actual_bonus_fights": [
                f"{r['fighter1']} vs {r['fighter2']}" for r in actual_bonus_fights
            ],
            "hit": local_hit,
        })

    n_events = len(per_event_records)
    hit_rate = hits / n_events if n_events else 0.0

    # Analytical random baseline: P(top-k random picks include a bonus)
    # = 1 - C(N-B, k) / C(N, k), averaged across events.
    def _p_random_hit(n: int, b: int, kk: int) -> float:
        if b == 0 or n == 0 or kk >= n:
            return 1.0 if b > 0 and kk >= n else 0.0
        if n - b < kk:
            return 1.0
        return 1.0 - math.comb(n - b, kk) / math.comb(n, kk)

    random_baseline = (
        sum(_p_random_hit(rec["num_fights"], rec["num_actual_bonus"], k)
            for rec in per_event_records) / n_events
        if n_events else 0.0
    )

    # ── Persist CSV ─────────────────────────────────────────────────────────
    with open(BACKTEST_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["event_date", "event_name", "num_fights", "num_actual_bonus",
             f"top_{k}_picks", "actual_bonus_fights", "hit"]
        )
        for rec in per_event_records:
            w.writerow([
                rec["event_date"],
                rec["event_name"],
                rec["num_fights"],
                rec["num_actual_bonus"],
                " | ".join(rec["top_k_picks"]),
                " | ".join(rec["actual_bonus_fights"]),
                int(rec["hit"]),
            ])

    # ── Persist Markdown table ─────────────────────────────────────────────
    with open(BACKTEST_MD, "w", encoding="utf-8") as fh:
        fh.write(f"# Backtest results — top-{k} per event (LogReg baseline)\n\n")
        fh.write(
            f"- Train cutoff: events before **{test_date_from}**\n"
            f"- Train size: **{len(train_rows)}** fights ({train_pos} positive, "
            f"{100*train_pos/len(train_rows):.1f}%)\n"
            f"- Test size: **{len(test_rows)}** fights across **{n_events}** events "
            f"({test_pos} positive)\n"
            f"- **Model top-{k} hit rate: {hit_rate:.0%}** "
            f"({hits}/{n_events} events had a bonus fight in the model's top {k})\n"
            f"- Random top-{k} hit rate (analytical baseline): "
            f"**{random_baseline:.0%}** — model uplift: "
            f"**{(hit_rate - random_baseline)*100:+.1f} pp**\n\n"
            f"> Caveat: only {n_events} events in the held-out 2026 test window. "
            "Per-event hit rate is noisy at this sample size; treat as illustrative.\n\n"
        )
        fh.write(f"| Date | Event | Top {k} picks | Actual bonus fights | Hit |\n")
        fh.write("|------|-------|---------------|--------------------|-----|\n")
        for rec in per_event_records:
            picks = "<br>".join(rec["top_k_picks"]) or "—"
            actual = "<br>".join(rec["actual_bonus_fights"]) or "—"
            hit = "✅" if rec["hit"] else "❌"
            fh.write(
                f"| {rec['event_date']} | {rec['event_name']} | "
                f"{picks} | {actual} | {hit} |\n"
            )

    logger.info("Wrote %s", BACKTEST_CSV)
    logger.info("Wrote %s", BACKTEST_MD)

    return {
        "test_date_from": test_date_from,
        "k": k,
        "train_fights": len(train_rows),
        "test_fights": len(test_rows),
        "events_tested": n_events,
        "events_with_hit": hits,
        "top_k_hit_rate": round(hit_rate, 3),
        "random_top_k_baseline": round(random_baseline, 3),
        "uplift_pp": round((hit_rate - random_baseline) * 100, 1),
        "actual_bonus_fights_in_test": total_actual_bonus_fights,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = backtest(Database())
    print(summary)
