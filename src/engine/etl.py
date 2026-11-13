"""
Medallion Architecture ETL & Star Schema Warehouse (DuckDB) for UFC Combat Telemetry.

Layers:
- bronze : Raw immutable landing tables (ingested from external CSV/Parquet or seeded fighter profiles)
- silver : Cleaned, deduplicated, type-cast, and domain-validated staging tables
- gold   : Star Schema analytical layer (dim_fighter, dim_bout, dim_round_interval, fact_interval_telemetry)
- main   : Backward-compatible views (fighter_profiles, interval_telemetry) for existing consumers
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

from src.engine.stats import FighterCombatProfile

DATA_PROVENANCE = "synthetic"  # updated dynamically to "cagequant_ingested" when real files are loaded


class OctagonDataEngine:
    """Embedded DuckDB Medallion + Star Schema warehouse for UFC fight telemetry."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.provenance = DATA_PROVENANCE
        self._initialize_medallion_schemas()

    def _initialize_medallion_schemas(self) -> None:
        """Creates bronze, silver, and gold schemas along with Star Schema dimensions and facts."""
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS bronze")
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS silver")
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS gold")

        # --- BRONZE LAYER (Raw Landing) ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS bronze.raw_fighters (
                fighter_name VARCHAR,
                weight_class VARCHAR,
                stance VARCHAR,
                punch_power DOUBLE,
                elbow_power DOUBLE,
                leg_kick_power DOUBLE,
                strike_hit_pct DOUBLE,
                arm_block_pct DOUBLE,
                leg_block_pct DOUBLE,
                takedown_def_pct DOUBLE,
                durability_index DOUBLE,
                takedown_success_pct DOUBLE,
                gnp_avg_damage DOUBLE,
                submission_conversion_pct DOUBLE,
                stamina_decay_rate DOUBLE,
                source_system VARCHAR,
                ingested_at TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS bronze.raw_bout_intervals (
                bout_id VARCHAR,
                interval_id INTEGER,
                fighter_name VARCHAR,
                opponent_name VARCHAR,
                round_num INTEGER,
                strikes_attempted INTEGER,
                strikes_landed INTEGER,
                damage_dealt DOUBLE,
                clinch_seconds INTEGER,
                ground_seconds INTEGER,
                opponent_td_attempted INTEGER,
                opponent_td_landed INTEGER,
                grapple_exchanges INTEGER,
                grapple_positions_won INTEGER,
                source_system VARCHAR,
                ingested_at TIMESTAMP
            )
        """)

        # --- SILVER LAYER (Cleaned & Validated Staging) ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS silver.stg_fighters (
                fighter_name VARCHAR PRIMARY KEY,
                weight_class VARCHAR,
                stance VARCHAR,
                punch_power DOUBLE,
                elbow_power DOUBLE,
                leg_kick_power DOUBLE,
                strike_hit_pct DOUBLE,
                arm_block_pct DOUBLE,
                leg_block_pct DOUBLE,
                takedown_def_pct DOUBLE,
                durability_index DOUBLE,
                takedown_success_pct DOUBLE,
                gnp_avg_damage DOUBLE,
                submission_conversion_pct DOUBLE,
                stamina_decay_rate DOUBLE,
                source_system VARCHAR,
                cleaned_at TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS silver.stg_bout_intervals (
                bout_id VARCHAR,
                interval_id INTEGER,
                fighter_name VARCHAR,
                opponent_name VARCHAR,
                round_num INTEGER,
                strikes_attempted INTEGER,
                strikes_landed INTEGER,
                damage_dealt DOUBLE,
                clinch_seconds INTEGER,
                ground_seconds INTEGER,
                opponent_td_attempted INTEGER,
                opponent_td_landed INTEGER,
                grapple_exchanges INTEGER,
                grapple_positions_won INTEGER,
                source_system VARCHAR,
                cleaned_at TIMESTAMP
            )
        """)

        # --- GOLD LAYER (Star Schema: Dimensions & Fact Table) ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS gold.dim_fighter (
                fighter_sk INTEGER PRIMARY KEY,
                fighter_name VARCHAR UNIQUE,
                weight_class VARCHAR,
                stance VARCHAR,
                punch_power DOUBLE,
                elbow_power DOUBLE,
                leg_kick_power DOUBLE,
                strike_hit_pct DOUBLE,
                arm_block_pct DOUBLE,
                leg_block_pct DOUBLE,
                takedown_def_pct DOUBLE,
                durability_index DOUBLE,
                takedown_success_pct DOUBLE,
                gnp_avg_damage DOUBLE,
                submission_conversion_pct DOUBLE,
                stamina_decay_rate DOUBLE,
                data_provenance VARCHAR
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS gold.dim_bout (
                bout_sk INTEGER PRIMARY KEY,
                bout_id VARCHAR UNIQUE,
                fighter_a_sk INTEGER,
                fighter_b_sk INTEGER,
                scheduled_rounds INTEGER,
                data_provenance VARCHAR
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS gold.dim_round_interval (
                interval_sk INTEGER PRIMARY KEY,
                round_num INTEGER,
                interval_in_round INTEGER,
                elapsed_seconds INTEGER
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS gold.fact_interval_telemetry (
                fact_id INTEGER PRIMARY KEY,
                bout_sk INTEGER,
                fighter_sk INTEGER,
                opponent_sk INTEGER,
                interval_sk INTEGER,
                interval_id INTEGER,
                round_num INTEGER,
                strikes_attempted INTEGER,
                strikes_landed INTEGER,
                damage_dealt DOUBLE,
                clinch_seconds INTEGER,
                ground_seconds INTEGER,
                opponent_td_attempted INTEGER,
                opponent_td_landed INTEGER,
                grapple_exchanges INTEGER,
                grapple_positions_won INTEGER
            )
        """)

        # --- BACKWARD-COMPATIBLE VIEWS IN MAIN SCHEMA ---
        self.conn.execute("""
            CREATE OR REPLACE VIEW fighter_profiles AS
            SELECT
                fighter_name, punch_power, elbow_power, leg_kick_power, strike_hit_pct,
                arm_block_pct, leg_block_pct, takedown_def_pct, durability_index,
                takedown_success_pct, gnp_avg_damage, submission_conversion_pct, stamina_decay_rate
            FROM gold.dim_fighter
        """)

        self.conn.execute("""
            CREATE OR REPLACE VIEW interval_telemetry AS
            SELECT
                f.interval_id,
                df.fighter_name,
                f.round_num,
                f.strikes_attempted,
                f.strikes_landed,
                f.damage_dealt,
                f.clinch_seconds,
                f.ground_seconds,
                f.opponent_td_attempted,
                f.opponent_td_landed,
                f.grapple_exchanges,
                f.grapple_positions_won
            FROM gold.fact_interval_telemetry f
            JOIN gold.dim_fighter df ON f.fighter_sk = df.fighter_sk
        """)

    # -------------------------------------------------------------------------
    # MEDALLION PIPELINE STAGES: BRONZE -> SILVER -> GOLD
    # -------------------------------------------------------------------------

    def ingest_bronze_records(
        self,
        fighters: List[tuple],
        intervals: List[tuple],
        source_system: str = "cagequant_calibrated_synthetic",
    ) -> None:
        """Loads raw fighter and interval records into the immutable Bronze layer."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        self.conn.execute("DELETE FROM bronze.raw_fighters")
        self.conn.execute("DELETE FROM bronze.raw_bout_intervals")

        fighter_rows = [(*row, source_system, now) for row in fighters]
        self.conn.executemany(
            "INSERT INTO bronze.raw_fighters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            fighter_rows,
        )

        interval_rows = [(*row, source_system, now) for row in intervals]
        self.conn.executemany(
            "INSERT INTO bronze.raw_bout_intervals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            interval_rows,
        )

    def ingest_bronze_from_files(
        self,
        fighters_csv: str | Path,
        intervals_csv: str | Path,
        source_system: str = "cagequant_external_export",
    ) -> None:
        """Data-engineers raw CSV/Parquet files from CageQuant into Bronze and runs Silver + Gold builds."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        self.provenance = "cagequant_ingested"
        self.conn.execute("DELETE FROM bronze.raw_fighters")
        self.conn.execute("DELETE FROM bronze.raw_bout_intervals")

        self.conn.execute(
            f"""
INSERT INTO bronze.raw_fighters
SELECT *, '{source_system}' AS source_system, TIMESTAMP '{now}' AS ingested_at
FROM read_csv_auto(?)
""",
[str(fighters_csv)],
)
self.conn.execute(
f"""
INSERT INTO bronze.raw_bout_intervals
SELECT *, '{source_system}' AS source_system, TIMESTAMP '{now}' AS ingested_at
FROM read_csv_auto(?)
""",
[str(intervals_csv)],
)
self.transform_bronze_to_silver()
self.build_gold_star_schema()
def transform_bronze_to_silver(self) -> None:
"""Deduplicates, clamps rates to, and enforces non-negative combat metrics into Silver."""
now = datetime.now(timezone.utc).replace(tzinfo=None)
self.conn.execute("DELETE FROM silver.stg_fighters")
self.conn.execute("DELETE FROM silver.stg_bout_intervals")
self.conn.execute(
f"""
INSERT INTO silver.stg_fighters
SELECT
TRIM(fighter_name) AS fighter_name,
COALESCE(TRIM(weight_class), 'Lightweight') AS weight_class,
COALESCE(TRIM(stance), 'Orthodox') AS stance,
GREATEST(0.0, punch_power) AS punch_power,
GREATEST(0.0, elbow_power) AS elbow_power,
GREATEST(0.0, leg_kick_power) AS leg_kick_power,
LEAST(1.0, GREATEST(0.0, strike_hit_pct)) AS strike_hit_pct,
LEAST(1.0, GREATEST(0.0, arm_block_pct)) AS arm_block_pct,
LEAST(1.0, GREATEST(0.0, leg_block_pct)) AS leg_block_pct,
LEAST(1.0, GREATEST(0.0, takedown_def_pct)) AS takedown_def_pct,
GREATEST(0.1, durability_index) AS durability_index,
LEAST(1.0, GREATEST(0.0, takedown_success_pct)) AS takedown_success_pct,
GREATEST(0.0, gnp_avg_damage) AS gnp_avg_damage,
LEAST(1.0, GREATEST(0.0, submission_conversion_pct)) AS submission_conversion_pct,
LEAST(1.0, GREATEST(0.001, stamina_decay_rate)) AS stamina_decay_rate,
source_system,
TIMESTAMP '{now}' AS cleaned_at
FROM (
SELECT *, ROW_NUMBER() OVER (PARTITION BY TRIM(fighter_name) ORDER BY ingested_at DESC) AS rn
FROM bronze.raw_fighters
WHERE fighter_name IS NOT NULL AND TRIM(fighter_name) <> ''
)
WHERE rn = 1
"""
)
self.conn.execute(
f"""
INSERT INTO silver.stg_bout_intervals
SELECT
TRIM(bout_id) AS bout_id,
CAST(interval_id AS INTEGER) AS interval_id,
TRIM(fighter_name) AS fighter_name,
TRIM(opponent_name) AS opponent_name,
GREATEST(1, CAST(round_num AS INTEGER)) AS round_num,
GREATEST(0, CAST(strikes_attempted AS INTEGER)) AS strikes_attempted,
LEAST(GREATEST(0, CAST(strikes_attempted AS INTEGER)), GREATEST(0, CAST(strikes_landed AS INTEGER))) AS strikes_landed,
GREATEST(0.0, damage_dealt) AS damage_dealt,
GREATEST(0, CAST(clinch_seconds AS INTEGER)) AS clinch_seconds,
GREATEST(0, CAST(ground_seconds AS INTEGER)) AS ground_seconds,
GREATEST(0, CAST(opponent_td_attempted AS INTEGER)) AS opponent_td_attempted,
LEAST(GREATEST(0, CAST(opponent_td_attempted AS INTEGER)), GREATEST(0, CAST(opponent_td_landed AS INTEGER))) AS opponent_td_landed,
GREATEST(0, CAST(grapple_exchanges AS INTEGER)) AS grapple_exchanges,
LEAST(GREATEST(0, CAST(grapple_exchanges AS INTEGER)), GREATEST(0, CAST(grapple_positions_won AS INTEGER))) AS grapple_positions_won,
source_system,
TIMESTAMP '{now}' AS cleaned_at
FROM bronze.raw_bout_intervals
WHERE fighter_name IS NOT NULL AND bout_id IS NOT NULL
"""
)
def build_gold_star_schema(self) -> None:
"""Builds Star Schema dimensions (dim_fighter, dim_bout, dim_round_interval) and fact_interval_telemetry."""
self.conn.execute("DELETE FROM gold.fact_interval_telemetry")
self.conn.execute("DELETE FROM gold.dim_round_interval")
self.conn.execute("DELETE FROM gold.dim_bout")
self.conn.execute("DELETE FROM gold.dim_fighter")
# 1. Populate dim_fighter
self.conn.execute("""
INSERT INTO gold.dim_fighter
SELECT
ROW_NUMBER() OVER (ORDER BY fighter_name) AS fighter_sk,
fighter_name, weight_class, stance,
punch_power, elbow_power, leg_kick_power, strike_hit_pct,
arm_block_pct, leg_block_pct, takedown_def_pct, durability_index,
takedown_success_pct, gnp_avg_damage, submission_conversion_pct, stamina_decay_rate,
source_system AS data_provenance
FROM silver.stg_fighters
""")
# 2. Populate dim_bout
self.conn.execute("""
INSERT INTO gold.dim_bout
SELECT
ROW_NUMBER() OVER (ORDER BY b.bout_id) AS bout_sk,
b.bout_id,
fa.fighter_sk AS fighter_a_sk,
fb.fighter_sk AS fighter_b_sk,
5 AS scheduled_rounds,
b.source_system AS data_provenance
FROM (
SELECT bout_id, MIN(fighter_name) AS fighter_a, MAX(opponent_name) AS fighter_b, MIN(source_system) AS source_system
FROM silver.stg_bout_intervals
GROUP BY bout_id
) b
LEFT JOIN gold.dim_fighter fa ON b.fighter_a = fa.fighter_name
LEFT JOIN gold.dim_fighter fb ON b.fighter_b = fb.fighter_name
""")
# 3. Populate dim_round_interval (5 rounds x 30 intervals of 10s = 150 intervals)
interval_dim_rows = []
sk = 1
for r in range(1, 6):
for idx in range(1, 31):
interval_dim_rows.append((sk, r, idx, ((r - 1) * 300) + (idx * 10)))
sk += 1
self.conn.executemany("INSERT INTO gold.dim_round_interval VALUES (?, ?, ?, ?)", interval_dim_rows)
# 4. Populate fact_interval_telemetry
self.conn.execute("""
INSERT INTO gold.fact_interval_telemetry
SELECT
ROW_NUMBER() OVER (ORDER BY s.bout_id, s.interval_id, s.fighter_name) AS fact_id,
db.bout_sk,
df.fighter_sk,
dof.fighter_sk AS opponent_sk,
dri.interval_sk,
s.interval_id,
s.round_num,
s.strikes_attempted,
s.strikes_landed,
s.damage_dealt,
s.clinch_seconds,
s.ground_seconds,
s.opponent_td_attempted,
s.opponent_td_landed,
s.grapple_exchanges,
s.grapple_positions_won
FROM silver.stg_bout_intervals s
JOIN gold.dim_bout db ON s.bout_id = db.bout_id
JOIN gold.dim_fighter df ON s.fighter_name = df.fighter_name
LEFT JOIN gold.dim_fighter dof ON s.opponent_name = dof.fighter_name
LEFT JOIN gold.dim_round_interval dri
ON s.round_num = dri.round_num AND ((s.interval_id - 1) % 30 + 1) = dri.interval_in_round
""")
# -------------------------------------------------------------------------
# SEEDING & QUERY INTERFACE
# -------------------------------------------------------------------------
def seed_simulated_telemetry(self) -> None:
"""Backward-compatible seeding method for test_metrics.py."""
self.seed_historical_lightweight_dataset()
def seed_historical_lightweight_dataset(self) -> None:
"""Seeds the 6 target fighters through Bronze -> Silver -> Gold Star Schema."""
# Profiles calibrated to UFC/CageQuant striking, grappling, defense & cardio tendencies
profiles_data = [
# (name, weight_class, stance, punch, elbow, leg_kick, hit_pct, arm_blk, leg_blk, td_def, durability, td_succ, gnp, sub_conv, stam_decay)
("Islam Makhachev", "Lightweight", "Southpaw", 24.0, 22.0, 14.0, 0.60, 0.48, 0.40, 0.90, 0.85, 0.62, 14.0, 0.25, 0.025),
("Carlos Prates", "Welterweight", "Southpaw", 26.0, 28.0, 18.0, 0.54, 0.35, 0.30, 0.55, 1.10, 0.42, 16.0, 0.35, 0.045),
("Justin Gaethje", "Lightweight", "Orthodox", 30.0, 18.0, 22.0, 0.49, 0.38, 0.25, 0.75, 0.90, 0.25, 10.0, 0.05, 0.040),
("Dustin Poirier", "Lightweight", "Southpaw", 28.0, 20.0, 16.0, 0.51, 0.42, 0.35, 0.64, 0.95, 0.36, 12.0, 0.12, 0.035),
("Charles Oliveira", "Lightweight", "Orthodox", 25.0, 24.0, 15.0, 0.53, 0.36, 0.32, 0.58, 0.88, 0.44, 15.0, 0.42, 0.038),
("Ilia Topuria", "Featherweight", "Orthodox", 31.0, 23.0, 17.0, 0.56, 0.46, 0.38, 0.92, 0.92, 0.56, 15.0, 0.28, 0.028),
]
telemetry_data = [
# Bout 1: Islam Makhachev vs Dustin Poirier
("UFC-302-MAIN", 1, "Islam Makhachev", "Dustin Poirier", 1, 8, 5, 14.0, 4, 6, 1, 0, 3, 3),
("UFC-302-MAIN", 2, "Islam Makhachev", "Dustin Poirier", 1, 10, 7, 18.0, 2, 8, 0, 0, 4, 3),
("UFC-302-MAIN", 1, "Dustin Poirier", "Islam Makhachev", 1, 11, 6, 15.0, 1, 0, 2, 1, 3, 0),
# Bout 2: Carlos Prates vs Justin Gaethje
("CQ-SIM-002", 1, "Carlos Prates", "Justin Gaethje", 1, 12, 6, 16.0, 1, 0, 2, 1, 3, 1),
("CQ-SIM-002", 2, "Carlos Prates", "Justin Gaethje", 1, 14, 7, 20.0, 0, 0, 1, 0, 4, 1),
("CQ-SIM-002", 1, "Justin Gaethje", "Carlos Prates", 1, 13, 7, 19.0, 0, 0, 0, 0, 2, 1),
# Bout 3: Charles Oliveira vs Ilia Topuria
("CQ-SIM-003", 1, "Charles Oliveira", "Ilia Topuria", 1, 9, 5, 14.5, 2, 3, 1, 0, 3, 2),
("CQ-SIM-003", 1, "Ilia Topuria", "Charles Oliveira", 1, 12, 7, 21.0, 1, 1, 1, 0, 3, 2),
]
self.ingest_bronze_records(profiles_data, telemetry_data, source_system="cagequant_calibrated_synthetic")
self.transform_bronze_to_silver()
self.build_gold_star_schema()
def load_fighter_profile(self, fighter_name: str) -> FighterCombatProfile:
"""Queries Gold dim_fighter for a fighter's profile and returns a typed FighterCombatProfile."""
row = self.conn.execute(
"""
SELECT fighter_name, punch_power, elbow_power, leg_kick_power, strike_hit_pct,
arm_block_pct, leg_block_pct, takedown_def_pct, durability_index,
takedown_success_pct, gnp_avg_damage, submission_conversion_pct, stamina_decay_rate
FROM gold.dim_fighter
WHERE fighter_name = ?
""",
[fighter_name],
).fetchone()
if not row:
raise ValueError(f"Fighter '{fighter_name}' not found in DuckDB database.")
return FighterCombatProfile(
name=row[0],
punch_power=row[1],
elbow_power=row[2],
leg_kick_power=row[3],
strike_hit_pct=row[4],
arm_block_pct=row[5],
leg_block_pct=row[6],
takedown_def_pct=row[7],
durability_index=row[8],
takedown_success_pct=row[9],
gnp_avg_damage=row[10],
submission_conversion_pct=row[11],
stamina_decay_rate=row[12],
)
def get_advanced_feature_metrics(self, fighter_name: str) -> Dict[str, float]:
"""Calculates advanced analytical metrics (xCT, Stamina Drain Ratio, Strike Volatility Index) from Gold fact table."""
query = """
SELECT
AVG(f.clinch_seconds + f.ground_seconds) / 10.0 AS xct,
AVG(f.damage_dealt) * 0.05 AS stamina_drain_ratio,
STDDEV_SAMP(f.damage_dealt) AS strike_volatility_idx
FROM gold.fact_interval_telemetry f
JOIN gold.dim_fighter df ON f.fighter_sk = df.fighter_sk
WHERE df.fighter_name = ?
"""
result = self.conn.execute(query, [fighter_name]).fetchone()
return {
"xct": round(result[0] or 0.0, 4),
"stamina_drain_ratio": round(result[1] or 0.0, 4),
"strike_volatility_idx": round(result[2] or 0.0, 4),
}
def get_fighter_pillar_metrics(self, fighter_name: str) -> Dict[str, float]:
"""Executes governed semantic queries over Gold Star Schema tables."""
query = """
SELECT
SUM(f.strikes_landed) * 1.0 / NULLIF(SUM(f.strikes_attempted), 0) AS strike_hit_pct,
SUM(f.damage_dealt) * 1.0 / NULLIF(SUM(f.strikes_landed), 0) AS avg_damage_per_strike,
1.0 - (SUM(f.opponent_td_landed) * 1.0 / NULLIF(SUM(f.opponent_td_attempted), 0)) AS takedown_def_pct,
SUM(f.grapple_positions_won) * 1.0 / NULLIF(SUM(f.grapple_exchanges), 0) AS grapple_win_pct
FROM gold.fact_interval_telemetry f
JOIN gold.dim_fighter df ON f.fighter_sk = df.fighter_sk
WHERE df.fighter_name = ?
"""
result = self.conn.execute(query, [fighter_name]).fetchone()
return {
"strike_hit_pct": round(result[0] or 0.50, 4) if result and result[0] is not None else 0.50,
"avg_damage_per_strike": round(result[1] or 10.0, 2) if result and result[1] is not None else 10.0,
"takedown_def_pct": round(result[2] or 0.75, 4) if result and result[2] is not None else 0.75,
"grapple_win_pct": round(result[3] or 0.50, 4) if result and result[3] is not None else 0.50,
}
def get_star_schema_summary(self) -> Dict[str, Any]:
"""Returns row counts across Bronze, Silver, and Gold Star Schema tables for observability."""
tables = [
"bronze.raw_fighters",
"bronze.raw_bout_intervals",
"silver.stg_fighters",
"silver.stg_bout_intervals",
"gold.dim_fighter",
"gold.dim_bout",
"gold.dim_round_interval",
"gold.fact_interval_telemetry",
]
counts = {}
for tbl in tables:
counts[tbl] = self.conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
return {"provenance": self.provenance, "table_counts": counts}
def get_training_frame(
self, n_fights: int = 1000, intervals_per_fight: int = 6, seed: int = 42
) -> pd.DataFrame:
"""Builds a SYNTHETIC grouped training frame sampled from Gold dim_fighter attributes."""
count = self.conn.execute("SELECT COUNT(*) FROM gold.dim_fighter").fetchone()[0]
if count == 0:
self.seed_historical_lightweight_dataset()
fighters_df = self.conn.execute(
"SELECT fighter_name, punch_power, strike_hit_pct, takedown_def_pct, durability_index, stamina_decay_rate FROM gold.dim_fighter"
).fetchdf()
rng = np.random.default_rng(seed)
rows = []
n_fighters = len(fighters_df)
for fight_id in range(1, n_fights + 1):
idx_a, idx_b = rng.choice(n_fighters, size=2, replace=False)
fa = fighters_df.iloc[idx_a]
fb = fighters_df.iloc[idx_b]
skill_diff = (
(fa["punch_power"] * fa["strike_hit_pct"] - fb["punch_power"] * fb["strike_hit_pct"]) * 0.18
+ (fa["takedown_def_pct"] - fb["takedown_def_pct"]) * 1.2
+ (fa["durability_index"] - fb["durability_index"]) * 0.8
)
logit = skill_diff + rng.normal(0.0, 0.45)
p_win = 1.0 / (1.0 + np.exp(-logit))
winner = int(rng.random() < p_win)
hp_a, hp_b = 100.0, 100.0
stam_a, stam_b = 100.0, 100.0
for step in range(1, intervals_per_fight + 1):
dmg_a = max(0.0, rng.normal(fa["punch_power"] * fa["strike_hit_pct"] * 0.55 + 1.2 * skill_diff, 2.5))
dmg_b = max(0.0, rng.normal(fb["punch_power"] * fb["strike_hit_pct"] * 0.55 - 1.2 * skill_diff, 2.5))
hp_a = max(0.0, hp_a - dmg_b)
hp_b = max(0.0, hp_b - dmg_a)
stam_a = max(10.0, stam_a - fa["stamina_decay_rate"] * 50.0)
stam_b = max(10.0, stam_b - fb["stamina_decay_rate"] * 50.0)
rows.append(
{
"fight_id": fight_id,
"round_num": (step - 1) // 2 + 1,
"fighter_a_hp": hp_a,
"fighter_b_hp": hp_b,
"fighter_a_stamina": stam_a,
"fighter_b_stamina": stam_b,
"damage_diff": dmg_a - dmg_b,
"power_diff": float(fa["punch_power"] - fb["punch_power"]),
"hp_delta": hp_a - hp_b,
"stamina_delta": stam_a - stam_b,
"bout_winner": winner,
}
)
return pd.DataFrame(rows)
