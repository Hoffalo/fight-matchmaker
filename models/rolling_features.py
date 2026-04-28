"""
models/rolling_features.py
Leak-safe rolling / recent form features from fight_stats for the UFC matchmaker.

Training: only fights with event_date strictly BEFORE the target fight date are used.

Inference: use ``get_inference_rolling_vector(..., before_date=None)`` to use all recorded
history through the latest event in the DB (or strictly before ``before_date`` if set).
"""
from __future__ import annotations

import logging
import sqlite3
import time
from datetime import date, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import BASE_DIR

logger = logging.getLogger(__name__)

ROLLING_FIGHTER_FEATURE_NAMES = [
    "recent_win_rate",
    "win_streak_norm",
    "momentum",
    "days_since_last_norm",
    "avg_days_between_norm",
    "recent_sig_strikes_pm_norm",
    "recent_sig_strike_acc",
    "recent_td_rate_norm",
    "recent_control_time_pct",
    "recent_knockdowns_norm",
    "strike_output_variance_norm",
    "performance_consistency",
    "strike_trend_norm",
    "damage_trend_norm",
    "n_prior_fights_norm",
]

ROLLING_MATCHUP_FEATURE_NAMES = [
    "variance_clash",
    "momentum_mismatch",
    "recent_output_combined",
    "trend_divergence",
]

ROLLING_DIM_FIGHTER = len(ROLLING_FIGHTER_FEATURE_NAMES)
ROLLING_DIM_MATCHUP = len(ROLLING_MATCHUP_FEATURE_NAMES)

CACHE_PATH = BASE_DIR / "data" / "rolling_features_cache.csv"
MOMENTUM_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.12, 0.08], dtype=np.float64)


def _norm(val: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def _parse_event_date(s: Any) -> Optional[pd.Timestamp]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = pd.to_datetime(str(s).strip(), errors="coerce")
    if pd.isna(t):
        return None
    return t


def get_fighter_fight_history(
    fighter_id: int,
    conn: sqlite3.Connection,
    before_date: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return fight_stats rows for ``fighter_id`` joined with fights/events, newest first.

    If ``before_date`` is set (ISO string), only rows with event_date < before_date.
    """
    q = """
    SELECT fs.*, f.winner_id, f.fighter1_id, f.fighter2_id, f.total_time_sec,
           e.date AS event_date
    FROM fight_stats fs
    JOIN fights f ON f.id = fs.fight_id
    JOIN events e ON e.id = f.event_id
    WHERE fs.fighter_id = ?
      AND e.date IS NOT NULL AND TRIM(e.date) != ''
    """
    params: list[Any] = [fighter_id]
    if before_date:
        q += " AND e.date < ?"
        params.append(before_date)
    q += " ORDER BY e.date DESC"
    cur = conn.execute(q, tuple(params))
    cols = [d[0] for d in cur.description]
    hist = [dict(zip(cols, row)) for row in cur.fetchall()]
    _enrich_history_opponent_sig_landed(conn, hist, fighter_id)
    return hist


def _enrich_history_opponent_sig_landed(
    conn: sqlite3.Connection,
    hist: list[dict[str, Any]],
    fighter_id: int,
) -> None:
    """Attach opponent sig strikes landed (damage proxy) for each row."""
    for h in hist:
        fid = h.get("fight_id")
        if fid is None:
            h["opp_sig_landed"] = 0.0
            continue
        row = conn.execute(
            """
            SELECT sig_strikes_landed FROM fight_stats
            WHERE fight_id = ? AND fighter_id != ?
            LIMIT 1
            """,
            (fid, fighter_id),
        ).fetchone()
        if row is None or row[0] is None:
            h["opp_sig_landed"] = 0.0
        else:
            h["opp_sig_landed"] = float(row[0])


def _fight_minutes(total_time_sec: Any) -> float:
    try:
        sec = float(total_time_sec or 0)
    except (TypeError, ValueError):
        sec = 0.0
    return max(sec / 60.0, 1e-3)


def _result_code(row: dict, fighter_id: int) -> float:
    w = row.get("winner_id")
    if w is None or w == "":
        return 0.5
    try:
        w = int(w)
    except (TypeError, ValueError):
        return 0.5
    if w == fighter_id:
        return 1.0
    return 0.0


def _career_fallback_vec(career: dict | None) -> np.ndarray:
    """Imputed rolling vector from fighter row aggregates when history < 2 fights."""
    c = career or {}
    sig_pm = float(c.get("sig_strikes_pm") or 3.0)
    sig_acc = float(c.get("sig_strike_acc") or 0.44)
    td_avg = float(c.get("td_avg") or 1.0)
    abs_pm = float(c.get("sig_strikes_abs_pm") or 3.0)
    tw = (c.get("wins_total") or 0) + (c.get("losses_total") or 0)
    win_pct = (c.get("wins_total") or 0) / max(tw, 1)
    ctrl_avg = float(c.get("ctrl_time_avg") or 60.0)

    return np.array(
        [
            win_pct,
            0.0,
            win_pct,
            0.5,
            0.5,
            _norm(sig_pm, 0.0, 12.0),
            max(0.0, min(1.0, sig_acc)),
            _norm(td_avg, 0.0, 8.0),
            _norm(ctrl_avg / 300.0, 0.0, 1.0),
            0.2,
            0.2,
            0.5,
            0.5,
            0.5,
            0.0,
        ],
        dtype=np.float32,
    )


def compute_rolling_features(
    fight_history: list[dict[str, Any]],
    fighter_id: int,
    asof_date: str,
    n_recent: int = 5,
    career_fallback: dict | None = None,
) -> np.ndarray:
    """
    Compute ``ROLLING_DIM_FIGHTER`` rolling features from prior fights (newest-first list).
    """
    asof = _parse_event_date(asof_date)
    if asof is None:
        asof = pd.Timestamp.now(tz=timezone.utc)

    if len(fight_history) < 2:
        return _career_fallback_vec(career_fallback)

    recent = fight_history[: min(n_recent, len(fight_history))]
    n = len(recent)
    n_prior = min(len(fight_history), n_recent)
    results = [_result_code(h, fighter_id) for h in recent]
    recent_win_rate = float(np.mean(results))

    streak = 0
    for r in results:
        if r >= 0.99:
            streak += 1
        elif r <= 0.01:
            break
        else:
            break
    win_streak_norm = min(streak, n_recent) / float(n_recent)

    wts = MOMENTUM_WEIGHTS[:n].copy()
    wts = wts / wts.sum()
    momentum = float(np.dot(np.array(results, dtype=np.float64), wts))

    last_dt = _parse_event_date(recent[0].get("event_date"))
    if last_dt is not None:
        days_since = (asof - last_dt).days
    else:
        days_since = 730
    days_since_last_norm = _norm(float(days_since), 0.0, 730.0)

    if n >= 2:
        dts = []
        for h in recent:
            t = _parse_event_date(h.get("event_date"))
            if t is not None:
                dts.append(t)
        dts_sorted = sorted(dts)
        gaps = [(dts_sorted[i] - dts_sorted[i - 1]).days for i in range(1, len(dts_sorted))]
        avg_gap = float(np.mean(gaps)) if gaps else 180.0
    else:
        avg_gap = 180.0
    avg_days_between_norm = _norm(avg_gap, 30.0, 400.0)

    sig_pms: list[float] = []
    accs: list[float] = []
    td_rates: list[float] = []
    ctrls: list[float] = []
    kds: list[float] = []
    damage = [float(h.get("opp_sig_landed") or 0) for h in recent]

    for h in recent:
        mins = _fight_minutes(h.get("total_time_sec"))
        landed = float(h.get("sig_strikes_landed") or 0)
        att = float(h.get("sig_strikes_att") or 0)
        sig_pms.append(landed / mins)
        accs.append(landed / max(att, 1.0))
        td_rates.append(float(h.get("td_landed") or 0) / mins * 15.0)
        tsec = float(h.get("total_time_sec") or 0)
        csec = float(h.get("ctrl_time_sec") or 0)
        ctrls.append(csec / max(tsec, 1.0))
        kds.append(float(h.get("knockdowns") or 0))

    recent_sig_pm = float(np.mean(sig_pms)) if sig_pms else 3.0
    recent_acc = float(np.mean(accs)) if accs else 0.44
    recent_td = float(np.mean(td_rates)) if td_rates else 1.0
    recent_ctrl = float(np.mean(ctrls)) if ctrls else 0.15
    recent_kd = float(np.mean(kds)) if kds else 0.0

    strike_var = float(np.std(sig_pms)) if len(sig_pms) > 1 else 0.0
    strike_var_norm = _norm(strike_var, 0.0, 5.0)

    total_sig = [float(h.get("sig_strikes_landed") or 0) for h in recent]
    mu = float(np.mean(total_sig)) if total_sig else 1.0
    sd = float(np.std(total_sig)) if len(total_sig) > 1 else 0.0
    cv = sd / max(mu, 1e-3)
    performance_consistency = max(0.0, min(1.0, 1.0 - _norm(cv, 0.0, 1.5)))

    xs = np.arange(len(sig_pms), dtype=np.float64)
    if len(sig_pms) >= 2:
        slope_strike, _ = np.polyfit(xs, np.array(sig_pms, dtype=np.float64), 1)
    else:
        slope_strike = 0.0
    strike_trend_norm = float(
        max(0.0, min(1.0, np.tanh(slope_strike / 3.0) * 0.5 + 0.5))
    )

    if len(damage) >= 2:
        slope_dmg, _ = np.polyfit(xs, np.array(damage, dtype=np.float64), 1)
    else:
        slope_dmg = 0.0
    damage_trend_norm = float(
        max(0.0, min(1.0, np.tanh(slope_dmg / 40.0) * 0.5 + 0.5))
    )

    n_prior_fights_norm = min(n_prior, n_recent) / float(n_recent)

    return np.array(
        [
            recent_win_rate,
            win_streak_norm,
            momentum,
            days_since_last_norm,
            avg_days_between_norm,
            _norm(recent_sig_pm, 0.0, 12.0),
            max(0.0, min(1.0, recent_acc)),
            _norm(recent_td, 0.0, 8.0),
            max(0.0, min(1.0, recent_ctrl)),
            _norm(recent_kd, 0.0, 3.0),
            strike_var_norm,
            performance_consistency,
            strike_trend_norm,
            damage_trend_norm,
            n_prior_fights_norm,
        ],
        dtype=np.float32,
    )


def compute_rolling_matchup_features(
    fighter_a_rolling: np.ndarray,
    fighter_b_rolling: np.ndarray,
) -> np.ndarray:
    """Four cross-features from aligned rolling vectors."""
    a, b = fighter_a_rolling, fighter_b_rolling
    variance_clash = float(np.clip(a[10] + b[10], 0.0, 2.0) / 2.0)
    momentum_mismatch = abs(float(a[2] - b[2]))
    recent_output_combined = float(np.clip(a[5] + b[5], 0.0, 2.0) / 2.0)
    trend_divergence = float(np.tanh(a[12] - b[12]) * 0.5 + 0.5)
    return np.array(
        [variance_clash, momentum_mismatch, recent_output_combined, trend_divergence],
        dtype=np.float32,
    )


def _add_opponent_sig_landed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    left = df[["fight_id", "fighter_id", "sig_strikes_landed"]].rename(
        columns={"sig_strikes_landed": "_sig_self"}
    )
    right = df[["fight_id", "fighter_id", "sig_strikes_landed"]].rename(
        columns={"fighter_id": "opp_fid", "sig_strikes_landed": "opp_sig_landed"}
    )
    comb = left.merge(right, on="fight_id")
    comb = comb[comb["fighter_id"] != comb["opp_fid"]]
    comb = comb.drop_duplicates(subset=["fight_id", "fighter_id"])
    return df.merge(
        comb[["fight_id", "fighter_id", "opp_sig_landed"]],
        on=["fight_id", "fighter_id"],
        how="left",
    ).assign(opp_sig_landed=lambda d: d["opp_sig_landed"].fillna(0.0))


def _load_stats_dataframe(db_path: str | Path) -> pd.DataFrame:
    db_path = str(db_path)
    q = """
    SELECT fs.fighter_id, fs.fight_id, fs.knockdowns, fs.sig_strikes_landed,
           fs.sig_strikes_att, fs.td_landed, fs.ctrl_time_sec,
           f.total_time_sec, f.winner_id, e.date AS event_date
    FROM fight_stats fs
    JOIN fights f ON f.id = fs.fight_id
    JOIN events e ON e.id = f.event_id
    WHERE e.date IS NOT NULL AND TRIM(e.date) != ''
    """
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(q, con)
    if df.empty:
        return df
    df["event_date_parsed"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date_parsed"])
    df = _add_opponent_sig_landed(df)
    return df.sort_values(["fighter_id", "event_date_parsed"])


def _rows_from_slice(g: pd.DataFrame, before_ts: pd.Timestamp) -> list[dict[str, Any]]:
    past = g[g["event_date_parsed"] < before_ts]
    if past.empty:
        return []
    past = past.sort_values("event_date_parsed", ascending=False)
    return past.head(30).to_dict("records")


def build_rolling_lookup_from_db(
    db_path: str | Path,
    fight_specs: list[tuple[int, int, int, str]],
    fighter_careers: dict[int, dict],
) -> dict[tuple[int, int], np.ndarray]:
    """One rolling vector per (fight_id, fighter_id)."""
    t0 = time.perf_counter()
    df = _load_stats_dataframe(db_path)
    lookup: dict[tuple[int, int], np.ndarray] = {}

    if df.empty:
        for fight_id, f1, f2, _ed in fight_specs:
            lookup[(fight_id, f1)] = _career_fallback_vec(fighter_careers.get(f1))
            lookup[(fight_id, f2)] = _career_fallback_vec(fighter_careers.get(f2))
        return lookup

    grouped = {k: v for k, v in df.groupby("fighter_id")}

    for fight_id, f1, f2, ed in fight_specs:
        asof = _parse_event_date(ed)
        if asof is None:
            asof = pd.Timestamp.now(tz=timezone.utc)
        ed_str = str(ed) if ed else str(asof.date())

        for oid in (f1, f2):
            g = grouped.get(oid)
            if g is None:
                lookup[(fight_id, oid)] = _career_fallback_vec(fighter_careers.get(oid))
                continue
            hist = _rows_from_slice(g, asof)
            lookup[(fight_id, oid)] = compute_rolling_features(
                hist,
                oid,
                ed_str,
                career_fallback=fighter_careers.get(oid),
            )

    logger.info(
        "Rolling features computed for %d fight corners in %.2fs",
        len(lookup),
        time.perf_counter() - t0,
    )
    return lookup


def _cache_is_fresh(cache_path: Path, db_path: Path) -> bool:
    if not cache_path.is_file() or not db_path.is_file():
        return False
    return cache_path.stat().st_mtime >= db_path.stat().st_mtime


def _lookup_to_cache_rows(
    lookup: dict[tuple[int, int], np.ndarray],
) -> pd.DataFrame:
    rows = []
    for (fid, oid), vec in lookup.items():
        row = {"fight_id": fid, "fighter_id": oid}
        for i, name in enumerate(ROLLING_FIGHTER_FEATURE_NAMES):
            row[name] = float(vec[i])
        rows.append(row)
    return pd.DataFrame(rows)


def _cache_rows_to_lookup(df: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    lookup: dict[tuple[int, int], np.ndarray] = {}
    for _, r in df.iterrows():
        vec = np.array(
            [float(r[nm]) for nm in ROLLING_FIGHTER_FEATURE_NAMES],
            dtype=np.float32,
        )
        lookup[(int(r["fight_id"]), int(r["fighter_id"]))] = vec
    return lookup


def get_rolling_lookup_cached(
    db_path: str | Path,
    fight_specs: list[tuple[int, int, int, str]],
    fighter_careers: dict[int, dict],
    cache_path: Path = CACHE_PATH,
    force_rebuild: bool = False,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Build or load rolling feature vectors for every (fight_id, corner) in fight_specs.
    Cache invalidates when the DB file is newer than the CSV.
    """
    db_path = Path(db_path)
    expected_keys = {(fs[0], fs[1]) for fs in fight_specs} | {(fs[0], fs[2]) for fs in fight_specs}

    if (
        not force_rebuild
        and _cache_is_fresh(cache_path, db_path)
    ):
        cached = pd.read_csv(cache_path)
        lookup = _cache_rows_to_lookup(cached)
        if set(lookup.keys()) == expected_keys:
            logger.info("Rolling features loaded from cache %s (%d rows)", cache_path, len(lookup))
            return lookup
        logger.info("Cache key mismatch; rebuilding rolling features")

    lookup = build_rolling_lookup_from_db(db_path, fight_specs, fighter_careers)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _lookup_to_cache_rows(lookup).to_csv(cache_path, index=False)
        logger.info("Wrote rolling feature cache to %s", cache_path)
    except OSError as e:
        logger.warning("Could not write rolling cache: %s", e)
    return lookup


def get_inference_rolling_vector(
    fighter_id: int,
    db_path: str | Path,
    before_date: str | None = None,
    career: dict | None = None,
) -> np.ndarray:
    """
    Rolling vector as of ``before_date`` (exclusive history). If ``before_date`` is None,
    use all fights in DB and measure recency vs **today** (matchmaking default).
    """
    db_path = Path(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        if before_date:
            hist = get_fighter_fight_history(fighter_id, conn, before_date=before_date)
            asof = before_date
        else:
            hist = get_fighter_fight_history(fighter_id, conn, before_date=None)
            asof = str(date.today())

    if not hist:
        return _career_fallback_vec(career)
    return compute_rolling_features(hist, fighter_id, asof, career_fallback=career)


def attach_rolling_to_fighter_dicts(
    db_path: str | Path,
    fight_rows: list[dict[str, Any]],
    fighters_map: dict[int, dict],
) -> dict[tuple[int, int], np.ndarray]:
    """Build lookup for fights with keys ``fight_id``, ``fighter1_id``, ``fighter2_id``, ``event_date``."""
    specs: list[tuple[int, int, int, str]] = []
    for fr in fight_rows:
        specs.append(
            (
                int(fr["fight_id"]),
                int(fr["fighter1_id"]),
                int(fr["fighter2_id"]),
                str(fr.get("event_date") or ""),
            )
        )
    careers = {fid: dict(f) for fid, f in fighters_map.items()}
    return get_rolling_lookup_cached(db_path, specs, careers)
