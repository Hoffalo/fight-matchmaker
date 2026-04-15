"""
data/db.py — SQLite database manager for UFC Matchmaker
"""
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from config import DB_PATH

logger = logging.getLogger(__name__)

SCHEMA = """
-- ── Fighters ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fighters (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL UNIQUE,
    nickname            TEXT,
    nationality         TEXT,
    birthdate           TEXT,
    height_cm           REAL,
    reach_cm            REAL,
    weight_lbs          REAL,
    weight_class        TEXT,
    stance              TEXT,

    -- Career aggregate stats
    wins_total          INTEGER DEFAULT 0,
    wins_ko             INTEGER DEFAULT 0,
    wins_sub            INTEGER DEFAULT 0,
    wins_dec            INTEGER DEFAULT 0,
    losses_total        INTEGER DEFAULT 0,
    losses_ko           INTEGER DEFAULT 0,
    losses_sub          INTEGER DEFAULT 0,
    losses_dec          INTEGER DEFAULT 0,
    draws               INTEGER DEFAULT 0,
    no_contests         INTEGER DEFAULT 0,

    -- UFC Stats per-minute averages
    sig_strikes_pm      REAL,   -- Significant strikes landed per minute
    sig_strike_acc      REAL,   -- Significant strike accuracy %
    sig_strikes_abs_pm  REAL,   -- Sig strikes absorbed per minute
    sig_strike_def      REAL,   -- Sig strike defense %
    td_avg              REAL,   -- Takedowns per 15 min
    td_acc              REAL,   -- Takedown accuracy %
    td_def              REAL,   -- Takedown defense %
    sub_avg             REAL,   -- Submission attempts per 15 min
    ctrl_time_avg       REAL,   -- Avg control time per fight (seconds)

    -- Computed style metrics
    ko_rate             REAL,   -- KO wins / total wins
    sub_rate            REAL,   -- Sub wins / total wins
    dec_rate            REAL,   -- Decision wins / total wins
    finish_rate         REAL,   -- (KO+Sub) / total wins
    grapple_ratio       REAL,   -- td_avg / (td_avg + sig_strikes_pm)

    -- Meta
    sherdog_url         TEXT,
    ufcstats_url        TEXT,
    tapology_url        TEXT,
    ranking             INTEGER,  -- Current UFC ranking in weight class (NULL = unranked)
    is_champion         INTEGER DEFAULT 0,
    last_updated        TEXT DEFAULT (datetime('now'))
);

-- ── Events ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    date        TEXT,
    location    TEXT,
    venue       TEXT,
    ufcstats_url TEXT UNIQUE,
    last_updated TEXT DEFAULT (datetime('now'))
);

-- ── Fights ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fights (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        INTEGER REFERENCES events(id),
    weight_class    TEXT,
    is_title_fight  INTEGER DEFAULT 0,
    is_main_event   INTEGER DEFAULT 0,

    -- Fighters
    fighter1_id     INTEGER REFERENCES fighters(id),
    fighter2_id     INTEGER REFERENCES fighters(id),
    winner_id       INTEGER REFERENCES fighters(id),

    -- Outcome
    method          TEXT,   -- KO/TKO, Submission, Decision - Unanimous, etc.
    method_detail   TEXT,   -- e.g. "Rear Naked Choke", "Punches"
    round           INTEGER,
    time            TEXT,   -- mm:ss of the round end
    total_time_sec  INTEGER, -- total fight duration in seconds

    -- Odds
    fighter1_odds   TEXT,
    fighter2_odds   TEXT,

    -- Fight-level quality metrics (computed post-scrape)
    total_sig_strikes INTEGER,
    sig_strikes_pm    REAL,
    total_tds         INTEGER,
    knockdowns        INTEGER,
    fight_quality_score REAL,  -- Our computed score

    ufcstats_url    TEXT UNIQUE,
    last_updated    TEXT DEFAULT (datetime('now'))
);

-- ── Fight Stats (per-fighter per-fight) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS fight_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fight_id        INTEGER REFERENCES fights(id),
    fighter_id      INTEGER REFERENCES fighters(id),

    -- Striking
    knockdowns          INTEGER DEFAULT 0,
    sig_strikes_landed  INTEGER DEFAULT 0,
    sig_strikes_att     INTEGER DEFAULT 0,
    total_strikes_landed INTEGER DEFAULT 0,
    total_strikes_att   INTEGER DEFAULT 0,

    -- Strikes by position (from UFCStats)
    head_landed     INTEGER DEFAULT 0,
    head_att        INTEGER DEFAULT 0,
    body_landed     INTEGER DEFAULT 0,
    body_att        INTEGER DEFAULT 0,
    leg_landed      INTEGER DEFAULT 0,
    leg_att         INTEGER DEFAULT 0,
    distance_landed INTEGER DEFAULT 0,
    distance_att    INTEGER DEFAULT 0,
    clinch_landed   INTEGER DEFAULT 0,
    clinch_att      INTEGER DEFAULT 0,
    ground_landed   INTEGER DEFAULT 0,
    ground_att      INTEGER DEFAULT 0,

    -- Grappling
    td_landed       INTEGER DEFAULT 0,
    td_att          INTEGER DEFAULT 0,
    sub_attempts    INTEGER DEFAULT 0,
    reversals       INTEGER DEFAULT 0,
    ctrl_time_sec   INTEGER DEFAULT 0,  -- Control time in seconds

    -- Computed share of fight action (for split analysis)
    sig_strike_share REAL,  -- This fighter's % of total sig strikes in fight
    td_share         REAL,
    ctrl_share       REAL
);

-- ── Indexes ────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_fights_fighter1 ON fights(fighter1_id);
CREATE INDEX IF NOT EXISTS idx_fights_fighter2 ON fights(fighter2_id);
CREATE INDEX IF NOT EXISTS idx_fight_stats_fight ON fight_stats(fight_id);
CREATE INDEX IF NOT EXISTS idx_fight_stats_fighter ON fight_stats(fighter_id);
CREATE INDEX IF NOT EXISTS idx_fighters_weight_class ON fighters(weight_class);
"""


class Database:
    """Thin wrapper around SQLite with context-manager support."""

    def __init__(self, path: str | None = None):
        self.path = path or str(DB_PATH)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self.connect() as conn:
            conn.executescript(SCHEMA)
        logger.info("Database ready at %s", self.path)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Fighters ─────────────────────────────────────────────────────────────

    def upsert_fighter(self, data: dict) -> int | None:
        """Insert or update a fighter. Returns fighter id."""
        cols = list(data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join([f"{c}=excluded.{c}" for c in cols if c != "name"])
        set_clause = (updates + ", " if updates else "") + "last_updated=datetime('now')"
        sql = (
            f"INSERT INTO fighters ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(name) DO UPDATE SET {set_clause}"
        )
        with self.connect() as conn:
            cursor = conn.execute(sql, list(data.values()))
            return cursor.lastrowid or self.get_fighter_id(data["name"])

    def get_fighter_id(self, name: str) -> int | None:
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM fighters WHERE name=?", (name,)).fetchone()
            return row["id"] if row else None

    def get_fighters_by_weight_class(self, weight_class: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fighters WHERE weight_class=? ORDER BY ranking ASC NULLS LAST",
                (weight_class,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_fighters(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM fighters").fetchall()
            return [dict(r) for r in rows]

    # ── Events ───────────────────────────────────────────────────────────────

    def upsert_event(self, data: dict) -> int | None:
        cols = list(data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join([f"{c}=excluded.{c}" for c in cols if c != "ufcstats_url"])
        set_clause = updates if updates else "ufcstats_url=excluded.ufcstats_url"
        sql = (
            f"INSERT INTO events ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(ufcstats_url) DO UPDATE SET {set_clause}"
        )
        with self.connect() as conn:
            cursor = conn.execute(sql, list(data.values()))
            return cursor.lastrowid or self._get_event_id(data["ufcstats_url"])

    def _get_event_id(self, url: str) -> int | None:
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM events WHERE ufcstats_url=?", (url,)).fetchone()
            return row["id"] if row else None

    def event_already_scraped(self, url: str) -> bool:
        """True if the event exists in DB and has at least one fight stored."""
        with self.connect() as conn:
            row = conn.execute(
                """SELECT e.id FROM events e
                   JOIN fights f ON f.event_id = e.id
                   WHERE e.ufcstats_url=? LIMIT 1""",
                (url,),
            ).fetchone()
            return row is not None

    def fighter_url_already_scraped(self, url: str) -> bool:
        """True if a fighter with this ufcstats_url is already in the DB."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM fighters WHERE ufcstats_url=?", (url,)
            ).fetchone()
            return row is not None

    # ── Fights ────────────────────────────────────────────────────────────────

    def upsert_fight(self, data: dict) -> int | None:
        cols = list(data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join([f"{c}=excluded.{c}" for c in cols if c != "ufcstats_url"])
        set_clause = updates if updates else "ufcstats_url=excluded.ufcstats_url"
        sql = (
            f"INSERT INTO fights ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(ufcstats_url) DO UPDATE SET {set_clause}"
        )
        with self.connect() as conn:
            cursor = conn.execute(sql, list(data.values()))
            return cursor.lastrowid or self._get_fight_id(data["ufcstats_url"])

    def _get_fight_id(self, url: str) -> int | None:
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM fights WHERE ufcstats_url=?", (url,)).fetchone()
            return row["id"] if row else None

    def insert_fight_stats(self, data: dict):
        cols = list(data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT OR REPLACE INTO fight_stats ({', '.join(cols)}) VALUES ({placeholders})"
        with self.connect() as conn:
            conn.execute(sql, list(data.values()))

    def get_fights_for_fighter(self, fighter_id: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT f.*, e.date, e.name as event_name
                   FROM fights f
                   JOIN events e ON f.event_id = e.id
                   WHERE f.fighter1_id=? OR f.fighter2_id=?
                   ORDER BY e.date DESC""",
                (fighter_id, fighter_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_fights(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT f.*, e.date, e.name as event_name
                   FROM fights f LEFT JOIN events e ON f.event_id = e.id"""
            ).fetchall()
            return [dict(r) for r in rows]

    def get_fight_stats(self, fight_id: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fight_stats WHERE fight_id=?", (fight_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with self.connect() as conn:
            fighters = conn.execute("SELECT COUNT(*) as n FROM fighters").fetchone()["n"]
            fights = conn.execute("SELECT COUNT(*) as n FROM fights").fetchone()["n"]
            events = conn.execute("SELECT COUNT(*) as n FROM events").fetchone()["n"]
            return {"fighters": fighters, "fights": fights, "events": events}
