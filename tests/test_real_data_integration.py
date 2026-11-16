"""Integration gate on the REAL snapshot. Skipped unless UFC_RAW_DIR points at the downloaded CSVs.

Used by the weekly ingest workflow: new data must pass these sanity contracts before it is committed.
"""
import os
from datetime import date
from pathlib import Path

import pytest

from src.engine.ingest import load_snapshot
from src.engine.medallion import UFCWarehouse

RAW_DIR = os.environ.get("UFC_RAW_DIR")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not RAW_DIR, reason="set UFC_RAW_DIR to run")]


@pytest.fixture(scope="module")
def wh():
    w = UFCWarehouse()
    w.load_bronze(load_snapshot(Path(RAW_DIR)))
    w.build_silver()
    w.build_gold(as_of=date.today())
    return w


def test_no_unknown_outcome_codes(wh):
    assert wh.query("SELECT count(*) n FROM stg_bouts WHERE result_code = 'UNKNOWN'").iloc[0]["n"] == 0


def test_bout_ids_are_unique(wh):
    d = wh.query("SELECT count(*) n, count(DISTINCT bout_id) u FROM stg_bouts").iloc[0]
    assert d["n"] == d["u"] and d["n"] > 8000


def test_undated_bouts_are_rare_and_excluded_from_gold(wh):
    dq = wh.data_quality()
    assert dq["undated_bouts"] / dq["bouts_total"] < 0.001
    assert wh.query("SELECT count(*) n FROM gold.fact_fighter_bouts WHERE event_date IS NULL").iloc[0]["n"] == 0


def test_stats_join_coverage_is_high(wh):
    raw = wh.query("SELECT count(*) n FROM raw_ufcstats_fight_stats").iloc[0]["n"]
    silver = wh.query("SELECT count(*) n FROM stg_round_stats").iloc[0]["n"]
    assert silver / raw > 0.99


def test_wins_equal_losses(wh):
    c = wh.query("SELECT result, count(*) n FROM gold.fact_fighter_bouts GROUP BY 1").set_index("result")["n"]
    assert c["WIN"] == c["LOSS"]


def test_rates_are_in_unit_interval(wh):
    assert wh.query("SELECT count(*) n FROM stg_round_stats WHERE sig_acc NOT BETWEEN 0 AND 1").iloc[0]["n"] == 0
