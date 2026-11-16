"""Slice 1 (RED): Bronze -> Silver -> Gold on tiny fixtures that mimic real UFCStats quirks."""
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.engine.ingest import load_snapshot
from src.engine.medallion import MIN_BOUTS_FOR_CARD, UFCWarehouse

FIXTURES = Path(__file__).parent / "fixtures" / "ufcstats"
INGESTED = datetime(2026, 11, 15, 8, 0, 0)


def build(as_of=date(2026, 1, 1), snapshot=None, loads=1):
    wh = UFCWarehouse()
    for _ in range(loads):
        wh.load_bronze(snapshot or load_snapshot(FIXTURES), ingested_at=INGESTED)
    wh.build_silver()
    wh.build_gold(as_of=as_of)
    return wh


def one(wh, sql, params=None):
    return wh.query(sql, params).iloc[0]


# ---------------- Bronze ----------------
def test_bronze_has_audit_columns_and_is_verbatim():
    wh = build()
    row = one(wh, "SELECT * FROM raw_ufcstats_fight_results ORDER BY URL LIMIT 1")
    assert row["source_system"] == "ufcstats_via_greco1899"
    assert row["ingested_at"] == INGESTED
    assert row["EVENT"] == "UFC 300: Test A "  # not cleaned in Bronze


def test_bronze_is_append_only_across_weekly_loads():
    wh = build(loads=2)
    assert one(wh, "SELECT count(*) n FROM raw_ufcstats_fight_results")["n"] == 10


# ---------------- Silver: bouts ----------------
def test_silver_dedups_repeated_snapshots_by_bout():
    wh = build(loads=2)
    n = one(wh, "SELECT count(*) n, count(DISTINCT bout_id) d FROM stg_bouts")
    assert n["n"] == 5 and n["d"] == 5


def test_silver_trims_event_and_joins_event_date():
    wh = build()
    row = one(wh, "SELECT event_name, event_date FROM stg_bouts WHERE bout_id = 'aaa1'")
    assert row["event_name"] == "UFC 300: Test A"
    assert pd.Timestamp(row["event_date"]).date() == date(2024, 1, 10)


@pytest.mark.parametrize(
    "bout_id, code, winner",
    [("aaa1", "A_WIN", "Alpha One"), ("bbb2", "B_WIN", "Charlie Three"), ("ddd4", "NC", None), ("eee5", "DRAW", None)],
)
def test_silver_resolves_winner_from_outcome_code(bout_id, code, winner):
    wh = build()
    row = one(wh, "SELECT result_code, winner FROM stg_bouts WHERE bout_id = ?", [bout_id])
    assert row["result_code"] == code
    if winner is None:
        assert pd.isna(row["winner"])
    else:
        assert row["winner"] == winner


@pytest.mark.parametrize("bout_id, seconds", [("aaa1", 390), ("bbb2", 900), ("ccc3", 250)])
def test_silver_total_fight_seconds(bout_id, seconds):
    wh = build()
    assert one(wh, "SELECT total_fight_sec t FROM stg_bouts WHERE bout_id = ?", [bout_id])["t"] == seconds


@pytest.mark.parametrize(
    "bout_id, wclass, gender, title, rounds",
    [("aaa1", "Lightweight", "M", False, 3), ("ccc3", "Lightweight", "M", True, 5), ("ddd4", "Strawweight", "F", False, 3)],
)
def test_silver_normalizes_weight_class_gender_title(bout_id, wclass, gender, title, rounds):
    wh = build()
    row = one(wh, "SELECT weight_class, gender, is_title, scheduled_rounds FROM stg_bouts WHERE bout_id = ?", [bout_id])
    assert (row["weight_class"], row["gender"], bool(row["is_title"]), row["scheduled_rounds"]) == (wclass, gender, title, rounds)


@pytest.mark.parametrize("bout_id, group", [("aaa1", "KO_TKO"), ("bbb2", "DEC"), ("ccc3", "SUB"), ("ddd4", "OTHER")])
def test_silver_method_groups(bout_id, group):
    wh = build()
    assert one(wh, "SELECT method_group g FROM stg_bouts WHERE bout_id = ?", [bout_id])["g"] == group


# ---------------- Silver: round stats ----------------
def test_silver_parses_of_strings_and_joins_to_bout_despite_whitespace():
    wh = build()
    row = one(wh, "SELECT sig_landed, sig_att, sig_acc, kd, ctrl_sec FROM stg_round_stats "
                  "WHERE bout_id = 'aaa1' AND fighter = 'Alpha One' AND round_no = 1")
    assert (row["sig_landed"], row["sig_att"], row["kd"], row["ctrl_sec"]) == (20, 30, 1, 60)
    assert row["sig_acc"] == pytest.approx(20 / 30)


def test_silver_missing_control_time_stays_null_not_zero():
    wh = build()
    row = one(wh, "SELECT ctrl_sec FROM stg_round_stats WHERE bout_id = 'aaa1' AND fighter = 'Alpha One' AND round_no = 2")
    assert pd.isna(row["ctrl_sec"])  # NULL, not 0


def test_silver_clamps_impossible_landed_greater_than_attempted():
    snap = load_snapshot(FIXTURES)
    stats = snap["fight_stats"].copy()
    stats.loc[(stats["FIGHTER"] == "Alpha One") & (stats["ROUND"] == "Round 1") & (stats["BOUT"] == "Alpha One vs. Bravo Two"), "SIG.STR."] = "50 of 40"
    wh = build(snapshot={**snap, "fight_stats": stats})
    row = one(wh, "SELECT sig_landed, sig_acc FROM stg_round_stats WHERE bout_id = 'aaa1' AND fighter = 'Alpha One' AND round_no = 1")
    assert row["sig_landed"] == 40 and row["sig_acc"] == 1.0


def test_all_silver_rates_are_within_unit_interval():
    wh = build()
    bad = one(wh, "SELECT count(*) n FROM stg_round_stats WHERE sig_acc < 0 OR sig_acc > 1 OR td_acc < 0 OR td_acc > 1")
    assert bad["n"] == 0


# ---------------- Silver: fighters ----------------
def test_silver_parses_physicals_and_flags_name_collisions():
    wh = build()
    alpha = one(wh, "SELECT height_in, reach_in, stance, dob, name_collision FROM stg_fighters WHERE fighter = 'Alpha One'")
    assert (alpha["height_in"], alpha["reach_in"], alpha["stance"]) == (70, 72, "Orthodox")
    assert pd.Timestamp(alpha["dob"]).date() == date(1995, 1, 1)
    assert not alpha["name_collision"]
    charlie = one(wh, "SELECT stance, height_in, name_collision FROM stg_fighters WHERE fighter = 'Charlie Three'")
    assert charlie["name_collision"]
    assert pd.isna(charlie["stance"]) and pd.isna(charlie["height_in"])  # never guess which Charlie it is


# ---------------- Gold ----------------
def test_gold_dim_fighter_career_aggregates():
    wh = build()
    row = one(wh, "SELECT ufc_bouts, ufc_minutes, last_fight_date, primary_weight_class, gender FROM gold.dim_fighter WHERE fighter = 'Alpha One'")
    assert row["ufc_bouts"] == 2
    assert row["ufc_minutes"] == pytest.approx((390 + 250) / 60)
    assert pd.Timestamp(row["last_fight_date"]).date() == date(2025, 6, 20)
    assert (row["primary_weight_class"], row["gender"]) == ("Lightweight", "M")


def test_gold_tracks_latest_weight_class_separately_from_most_common():
    snap = load_snapshot(FIXTURES)
    results = snap["fight_results"].copy()
    results.loc[results["BOUT"] == "Bravo Two vs. Delta Four", "WEIGHTCLASS"] = "Welterweight Bout"  # Bravo's latest bout
    wh = build(snapshot={**snap, "fight_results": results})
    row = one(wh, "SELECT primary_weight_class p, latest_weight_class l FROM gold.dim_fighter WHERE fighter = 'Bravo Two'")
    assert row["l"] == "Welterweight"
    assert row["p"] in ("Lightweight", "Welterweight")  # tie between two bouts; latest is the one we rely on


def test_gold_provisional_flag_uses_minimum_bouts_floor():
    assert MIN_BOUTS_FOR_CARD == 7
    wh = build()
    assert wh.query("SELECT bool_and(is_provisional) b FROM gold.dim_fighter").iloc[0]["b"]


@pytest.mark.parametrize(
    "as_of, status",
    [(date(2025, 12, 31), "ACTIVE"), (date(2026, 7, 1), "STALE"), (date(2028, 1, 1), "INACTIVE")],
)
def test_gold_activity_status_thresholds(as_of, status):
    wh = build(as_of=as_of)
    assert one(wh, "SELECT status FROM gold.dim_fighter WHERE fighter = 'Alpha One'")["status"] == status


def test_gold_fact_is_point_in_time():
    wh = build()
    df = wh.query("SELECT bout_id, prior_ufc_bouts, prior_ufc_seconds FROM gold.fact_fighter_bouts "
                  "WHERE fighter = 'Alpha One' ORDER BY event_date")
    assert df["prior_ufc_bouts"].tolist() == [0, 1]
    assert df["prior_ufc_seconds"].tolist() == [0, 390]


def test_gold_fact_aggregates_rounds_and_mirrors_opponent():
    wh = build()
    row = one(wh, "SELECT kd, sig_landed, sig_absorbed, result, opponent, has_round_stats FROM gold.fact_fighter_bouts "
                  "WHERE bout_id = 'aaa1' AND fighter = 'Alpha One'")
    assert (row["kd"], row["sig_landed"], row["sig_absorbed"]) == (2, 28, 12)
    assert (row["result"], row["opponent"], bool(row["has_round_stats"])) == ("WIN", "Bravo Two", True)


def test_gold_fact_marks_bouts_without_round_stats():
    wh = build()
    assert not one(wh, "SELECT has_round_stats h FROM gold.fact_fighter_bouts WHERE bout_id = 'bbb2' LIMIT 1")["h"]


def test_gold_wins_equal_losses_and_nc_draws_are_kept():
    wh = build()
    c = wh.query("SELECT result, count(*) n FROM gold.fact_fighter_bouts GROUP BY 1").set_index("result")["n"].to_dict()
    assert c["WIN"] == c["LOSS"] == 3 and c["NC"] == 2 and c["DRAW"] == 2


def test_gold_dim_bout_one_row_per_bout():
    wh = build()
    assert one(wh, "SELECT count(*) n, count(DISTINCT bout_id) d FROM gold.dim_bout")["n"] == 5


# ---------------- Data quality: undated bouts ----------------
def test_undated_bouts_stay_in_silver_but_never_reach_gold():
    snap = load_snapshot(FIXTURES)
    events = snap["events"]
    wh = build(snapshot={**snap, "events": events[events["EVENT"] != "UFC Fight Night: Test C"]})
    assert one(wh, "SELECT count(*) n FROM stg_bouts WHERE event_date IS NULL")["n"] == 1
    assert one(wh, "SELECT count(*) n FROM gold.dim_bout")["n"] == 4
    assert one(wh, "SELECT count(*) n FROM gold.fact_fighter_bouts WHERE event_date IS NULL")["n"] == 0
    assert wh.data_quality()["undated_bouts"] == 1


def test_data_quality_summary_on_clean_fixtures():
    dq = build().data_quality()
    assert dq == {"bouts_total": 5, "undated_bouts": 0, "unknown_outcomes": 0, "unjoined_stats_rows": 0}
