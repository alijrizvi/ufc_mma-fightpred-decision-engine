"""Pure DuckDB Medallion warehouse (Bronze -> Silver -> Gold) for real UFCStats data.

Layers:
  Bronze  raw_ufcstats_*           verbatim, append-only weekly snapshots + audit columns
  Silver  stg_bouts / stg_round_stats / stg_fighters
                                   deduplicated (window functions), typed, trimmed, clamped
  Gold    gold.dim_fighter / gold.dim_bout / gold.fact_fighter_bouts
                                   star schema; fact rows carry point-in-time (pre-fight) columns

Known limitation (ADR 015/019): UFCStats stats rows carry fighter *names*, not IDs, so fighters are
name-keyed. Names that appear more than once upstream are flagged ``name_collision`` and their
physical attributes are nulled rather than guessed.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, Optional, Sequence

import duckdb
import pandas as pd

from src.engine.ingest import SOURCE_SYSTEM

MIN_BOUTS_FOR_CARD = 7        # ADR 019: fewer UFC bouts -> provisional card
STALE_AFTER_DAYS = 365        # ADR 019: no fight for 12 months -> STALE flag
INACTIVE_AFTER_DAYS = 730     # ADR 019: no fight for 24 months -> INACTIVE (out of percentile population)

_BRONZE_TABLES = {
    "fight_results": "raw_ufcstats_fight_results",
    "fight_stats": "raw_ufcstats_fight_stats",
    "fighters": "raw_ufcstats_fighters",
    "events": "raw_ufcstats_events",
}

# (source column, silver prefix) pairs stored upstream as "<landed> of <attempted>"
_OF_COLUMNS = [
    ("SIG.STR.", "sig"), ("TOTAL STR.", "total"), ("TD", "td"), ("HEAD", "head"), ("BODY", "body"),
    ("LEG", "leg"), ("DISTANCE", "distance"), ("CLINCH", "clinch"), ("GROUND", "ground"),
]

_MACROS = r"""
CREATE OR REPLACE MACRO of_landed(x) AS try_cast(regexp_extract(trim(CAST(x AS VARCHAR)), '^(\d+) of (\d+)$', 1) AS INTEGER);
CREATE OR REPLACE MACRO of_att(x)    AS try_cast(regexp_extract(trim(CAST(x AS VARCHAR)), '^(\d+) of (\d+)$', 2) AS INTEGER);
CREATE OR REPLACE MACRO mmss_sec(x)  AS CASE WHEN regexp_matches(trim(CAST(x AS VARCHAR)), '^\d+:\d{2}$')
    THEN CAST(split_part(trim(CAST(x AS VARCHAR)), ':', 1) AS INTEGER) * 60
       + CAST(split_part(trim(CAST(x AS VARCHAR)), ':', 2) AS INTEGER) END;
"""


def _round_stats_sql() -> str:
    parsed = ",\n    ".join(
        f'of_landed("{src}") AS {p}_l_raw, of_att("{src}") AS {p}_att' for src, p in _OF_COLUMNS
    )
    clamped = ",\n    ".join(
        f"LEAST({p}_l_raw, {p}_att) AS {p}_landed, {p}_att,\n"
        f"    CASE WHEN {p}_att > 0 THEN CAST(LEAST({p}_l_raw, {p}_att) AS DOUBLE) / {p}_att END AS {p}_acc"
        for _, p in _OF_COLUMNS
    )
    return f"""
CREATE OR REPLACE TABLE stg_round_stats AS
WITH dedup AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY trim("EVENT"), trim("BOUT"), "ROUND", trim("FIGHTER") ORDER BY ingested_at DESC) AS rn
    FROM raw_ufcstats_fight_stats
), keys AS (
    SELECT bout_id, event_name, bout_name FROM stg_bouts
    QUALIFY ROW_NUMBER() OVER (PARTITION BY event_name, bout_name ORDER BY bout_id) = 1
), parsed AS (
    SELECT k.bout_id,
           try_cast(regexp_extract(CAST(d."ROUND" AS VARCHAR), '(\\d+)', 1) AS INTEGER) AS round_no,
           trim(d."FIGHTER") AS fighter,
           try_cast(d."KD" AS INTEGER) AS kd,
           try_cast(d."SUB.ATT" AS INTEGER) AS sub_att,
           try_cast(d."REV." AS INTEGER) AS rev,
           mmss_sec(d."CTRL") AS ctrl_sec,
           {parsed},
           d.ingested_at
    FROM dedup d JOIN keys k ON trim(d."EVENT") = k.event_name AND trim(d."BOUT") = k.bout_name
    WHERE d.rn = 1
)
SELECT bout_id, round_no, fighter, kd, sub_att, rev, ctrl_sec,
    {clamped},
    ingested_at
FROM parsed;
"""


class UFCWarehouse:
    """DuckDB warehouse implementing the Medallion layers for UFCStats data."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.conn = duckdb.connect(db_path)
        self.conn.execute(_MACROS)
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS gold;")

    # ------------------------------------------------------------------ helpers
    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> pd.DataFrame:
        """Run a read query and return a DataFrame."""
        return self.conn.execute(sql, list(params or [])).df()

    # ------------------------------------------------------------------ Bronze
    def load_bronze(
        self,
        snapshot: Dict[str, pd.DataFrame],
        source_system: str = SOURCE_SYSTEM,
        ingested_at: Optional[datetime] = None,
    ) -> None:
        """Append a verbatim weekly snapshot with ``source_system`` and ``ingested_at`` audit columns."""
        stamp = ingested_at or datetime.now(timezone.utc).replace(tzinfo=None)
        for name, table in _BRONZE_TABLES.items():
            frame = snapshot[name].copy()
            frame["source_system"] = source_system
            frame["ingested_at"] = stamp
            self.conn.register("_incoming", frame)
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM _incoming WHERE 1=0")
            self.conn.execute(f"INSERT INTO {table} SELECT * FROM _incoming")
            self.conn.unregister("_incoming")

    # ------------------------------------------------------------------ Silver
    def build_silver(self) -> None:
        """Rebuild Silver tables from Bronze (idempotent)."""
        self.conn.execute(self._stg_bouts_sql())
        self.conn.execute(_round_stats_sql())
        self.conn.execute(self._stg_fighters_sql())

    @staticmethod
    def _stg_bouts_sql() -> str:
        return r"""
CREATE OR REPLACE TABLE stg_bouts AS
WITH latest AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY "URL" ORDER BY ingested_at DESC) AS rn
    FROM raw_ufcstats_fight_results
), ev AS (
    SELECT trim("EVENT") AS event_name, try_strptime(trim("DATE"), '%B %d, %Y')::DATE AS event_date,
           ROW_NUMBER() OVER (PARTITION BY trim("EVENT") ORDER BY ingested_at DESC) AS rn
    FROM raw_ufcstats_events
), base AS (
    SELECT regexp_extract("URL", '[^/]+$') AS bout_id,
           trim(l."EVENT") AS event_name,
           trim(l."BOUT") AS bout_name,
           trim(split_part(l."BOUT", ' vs. ', 1)) AS fighter_a,
           trim(split_part(l."BOUT", ' vs. ', 2)) AS fighter_b,
           trim(l."OUTCOME") AS outcome,
           trim(l."WEIGHTCLASS") AS weightclass_raw,
           trim(l."METHOD") AS method_detail,
           try_cast(l."ROUND" AS INTEGER) AS end_round,
           mmss_sec(l."TIME") AS end_time_sec,
           trim(l."TIME FORMAT") AS time_format,
           l.ingested_at
    FROM latest l WHERE l.rn = 1
)
SELECT b.bout_id, b.event_name, e.event_date, b.bout_name, b.fighter_a, b.fighter_b,
       CASE b.outcome WHEN 'W/L' THEN 'A_WIN' WHEN 'L/W' THEN 'B_WIN' WHEN 'D/D' THEN 'DRAW'
                      WHEN 'NC/NC' THEN 'NC' ELSE 'UNKNOWN' END AS result_code,
       CASE b.outcome WHEN 'W/L' THEN b.fighter_a WHEN 'L/W' THEN b.fighter_b END AS winner,
       CASE WHEN lower(b.weightclass_raw) LIKE '%light heavyweight%' THEN 'Light Heavyweight'
            WHEN lower(b.weightclass_raw) LIKE '%super heavyweight%' THEN 'Super Heavyweight'
            WHEN lower(b.weightclass_raw) LIKE '%heavyweight%' THEN 'Heavyweight'
            WHEN lower(b.weightclass_raw) LIKE '%middleweight%' THEN 'Middleweight'
            WHEN lower(b.weightclass_raw) LIKE '%welterweight%' THEN 'Welterweight'
            WHEN lower(b.weightclass_raw) LIKE '%lightweight%' THEN 'Lightweight'
            WHEN lower(b.weightclass_raw) LIKE '%featherweight%' THEN 'Featherweight'
            WHEN lower(b.weightclass_raw) LIKE '%bantamweight%' THEN 'Bantamweight'
            WHEN lower(b.weightclass_raw) LIKE '%flyweight%' THEN 'Flyweight'
            WHEN lower(b.weightclass_raw) LIKE '%strawweight%' THEN 'Strawweight'
            ELSE 'Open/Catch Weight' END AS weight_class,
       CASE WHEN lower(b.weightclass_raw) LIKE '%women%' THEN 'F' ELSE 'M' END AS gender,
       (lower(b.weightclass_raw) LIKE '%title%' OR lower(b.weightclass_raw) LIKE '%championship%') AS is_title,
       try_cast(regexp_extract(b.time_format, '^(\d+) Rnd', 1) AS INTEGER) AS scheduled_rounds,
       CASE WHEN lower(b.method_detail) LIKE 'ko/tko%' OR lower(b.method_detail) LIKE 'tko%' THEN 'KO_TKO'
            WHEN lower(b.method_detail) LIKE 'submission%' THEN 'SUB'
            WHEN lower(b.method_detail) LIKE 'decision%' THEN 'DEC'
            ELSE 'OTHER' END AS method_group,
       b.method_detail, b.end_round, b.end_time_sec,
       (b.end_round - 1) * 300 + b.end_time_sec AS total_fight_sec,
       b.time_format, b.ingested_at
FROM base b LEFT JOIN (SELECT * FROM ev WHERE rn = 1) e ON b.event_name = e.event_name;
"""

    @staticmethod
    def _stg_fighters_sql() -> str:
        return r"""
CREATE OR REPLACE TABLE stg_fighters AS
WITH latest AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY "URL" ORDER BY ingested_at DESC) AS rn
    FROM raw_ufcstats_fighters
), named AS (
    SELECT trim("FIGHTER") AS fighter, "HEIGHT" AS h, "REACH" AS r, nullif(trim("STANCE"), '') AS stance_raw,
           try_strptime(trim("DOB"), '%b %d, %Y')::DATE AS dob_raw, "URL" AS fighter_url,
           count(*) OVER (PARTITION BY trim("FIGHTER")) > 1 AS name_collision,
           ROW_NUMBER() OVER (PARTITION BY trim("FIGHTER") ORDER BY "URL") AS name_rank
    FROM latest WHERE rn = 1
)
SELECT fighter,
       CASE WHEN name_collision THEN NULL ELSE
            try_cast(regexp_extract(h, '(\d+)''', 1) AS INTEGER) * 12
            + coalesce(try_cast(regexp_extract(h, '(\d+)"', 1) AS INTEGER), 0) END AS height_in,
       CASE WHEN name_collision THEN NULL ELSE try_cast(regexp_extract(r, '(\d+)', 1) AS INTEGER) END AS reach_in,
       CASE WHEN name_collision THEN NULL ELSE stance_raw END AS stance,
       CASE WHEN name_collision THEN NULL ELSE dob_raw END AS dob,
       name_collision,
       CASE WHEN name_collision THEN NULL ELSE fighter_url END AS fighter_url
FROM named WHERE name_rank = 1;
"""

    def data_quality(self) -> Dict[str, int]:
        """Silver-layer data-quality counters (surfaced by CI and the weekly ingest job)."""
        row = self.query("""
            SELECT (SELECT count(*) FROM stg_bouts) AS bouts_total,
                   (SELECT count(*) FROM stg_bouts WHERE event_date IS NULL) AS undated_bouts,
                   (SELECT count(*) FROM stg_bouts WHERE result_code = 'UNKNOWN') AS unknown_outcomes,
                   (SELECT count(*) FROM (SELECT DISTINCT trim("EVENT"), trim("BOUT"), "ROUND", trim("FIGHTER")
                                          FROM raw_ufcstats_fight_stats)) - (SELECT count(*) FROM stg_round_stats)
                       AS unjoined_stats_rows
        """).iloc[0]
        return {k: int(row[k]) for k in row.index}

    # ------------------------------------------------------------------ Gold
    def build_gold(self, as_of: Optional[date] = None) -> None:
        """Rebuild the Gold star schema. ``as_of`` fixes the clock for STALE/INACTIVE status (testable)."""
        as_of = as_of or date.today()
        self.conn.execute("""
CREATE OR REPLACE TABLE gold.dim_bout AS
SELECT bout_id, event_name, event_date, bout_name, weight_class, gender, is_title, scheduled_rounds,
       method_group, method_detail, end_round, total_fight_sec, result_code, winner
FROM stg_bouts WHERE event_date IS NOT NULL;
""")
        self.conn.execute(r"""
CREATE OR REPLACE TABLE gold.fact_fighter_bouts AS
WITH sides AS (
    SELECT bout_id, event_date, fighter_a AS fighter, fighter_b AS opponent, 'A' AS side, result_code,
           method_group, total_fight_sec, weight_class, gender FROM stg_bouts WHERE event_date IS NOT NULL
    UNION ALL
    SELECT bout_id, event_date, fighter_b, fighter_a, 'B', result_code,
           method_group, total_fight_sec, weight_class, gender FROM stg_bouts WHERE event_date IS NOT NULL
), agg AS (
    SELECT bout_id, fighter, count(*) AS rounds_with_stats,
           sum(kd) AS kd, sum(sub_att) AS sub_att, sum(rev) AS rev, sum(ctrl_sec) AS ctrl_sec,
           sum(sig_landed) AS sig_landed, sum(sig_att) AS sig_att,
           sum(td_landed) AS td_landed, sum(td_att) AS td_att,
           sum(head_landed) AS head_landed, sum(distance_landed) AS distance_landed,
           sum(clinch_landed) AS clinch_landed, sum(ground_landed) AS ground_landed
    FROM stg_round_stats GROUP BY bout_id, fighter
)
SELECT s.bout_id, s.event_date, s.fighter, s.opponent, s.side,
       CASE WHEN s.result_code = 'DRAW' THEN 'DRAW' WHEN s.result_code IN ('NC', 'UNKNOWN') THEN 'NC'
            WHEN (s.result_code = 'A_WIN' AND s.side = 'A') OR (s.result_code = 'B_WIN' AND s.side = 'B') THEN 'WIN'
            ELSE 'LOSS' END AS result,
       s.method_group, s.total_fight_sec, s.weight_class, s.gender,
       a.fighter IS NOT NULL AS has_round_stats,
       a.kd, a.sub_att, a.rev, a.ctrl_sec, a.sig_landed, a.sig_att, a.td_landed, a.td_att,
       a.head_landed, a.distance_landed, a.clinch_landed, a.ground_landed,
       o.sig_landed AS sig_absorbed, o.sig_att AS sig_att_faced, o.td_landed AS td_conceded, o.td_att AS td_faced,
       o.kd AS kd_suffered,
       ROW_NUMBER() OVER w - 1 AS prior_ufc_bouts,
       coalesce(sum(s.total_fight_sec) OVER (PARTITION BY s.fighter ORDER BY s.event_date, s.bout_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS prior_ufc_seconds
FROM sides s
LEFT JOIN agg a ON a.bout_id = s.bout_id AND a.fighter = s.fighter
LEFT JOIN agg o ON o.bout_id = s.bout_id AND o.fighter = s.opponent
WINDOW w AS (PARTITION BY s.fighter ORDER BY s.event_date, s.bout_id);
""")
        self.conn.execute(
            rf"""
CREATE OR REPLACE TABLE gold.dim_fighter AS
WITH career AS (
    SELECT fighter, count(*) AS ufc_bouts, sum(total_fight_sec) / 60.0 AS ufc_minutes,
           min(event_date) AS first_fight_date, max(event_date) AS last_fight_date,
           mode(weight_class) AS primary_weight_class, arg_max(weight_class, event_date) AS latest_weight_class,
           mode(gender) AS gender
    FROM gold.fact_fighter_bouts GROUP BY fighter
)
SELECT c.fighter, c.gender, c.primary_weight_class, c.latest_weight_class, c.ufc_bouts, c.ufc_minutes,
       c.first_fight_date, c.last_fight_date,
       date_diff('day', c.last_fight_date, ?::DATE) AS days_since_last_fight,
       CASE WHEN date_diff('day', c.last_fight_date, ?::DATE) > {INACTIVE_AFTER_DAYS} THEN 'INACTIVE'
            WHEN date_diff('day', c.last_fight_date, ?::DATE) > {STALE_AFTER_DAYS} THEN 'STALE'
            ELSE 'ACTIVE' END AS status,
       c.ufc_bouts < {MIN_BOUTS_FOR_CARD} AS is_provisional,
       f.height_in, f.reach_in, f.stance, f.dob, coalesce(f.name_collision, FALSE) AS name_collision
FROM career c LEFT JOIN stg_fighters f ON f.fighter = c.fighter;
""",
            [as_of, as_of, as_of],
        )
