"""Slice 2 (RED): Bayesian shrinkage + point-in-time (PIT) feature counts on tiny fixtures."""
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from src.engine.features import (
    ATTRIBUTES,
    FEATURES,
    METRICS,
    build_pit_counts,
    compute_features,
    fit_priors,
    shrink_rate,
)
from src.engine.ingest import load_snapshot
from src.engine.medallion import UFCWarehouse

FIXTURES = Path(__file__).parent / "fixtures" / "ufcstats"
AS_OF = date(2026, 1, 1)


@pytest.fixture(scope="module")
def counts():
    wh = UFCWarehouse()
    wh.load_bronze(load_snapshot(FIXTURES), ingested_at=datetime(2026, 11, 15))
    wh.build_silver()
    wh.build_gold(as_of=AS_OF)
    return build_pit_counts(wh, as_of=AS_OF)


def row(df, fighter, bout_id):
    return df[(df["fighter"] == fighter) & (df["bout_id"] == bout_id)].iloc[0]


# ---------------- shrinkage ----------------
def test_shrink_rate_matches_closed_form():
    # (M * raw + k * mu) / (M + k) with raw = 20/10 = 2.0, mu = 1.0, k = 30  ->  (20 + 30) / 40
    assert shrink_rate(20, 10, 1.0, 30) == pytest.approx(1.25)


def test_shrink_rate_zero_exposure_returns_prior():
    assert shrink_rate(0, 0, 0.42, 30) == pytest.approx(0.42)


def test_shrink_rate_huge_exposure_returns_raw():
    assert shrink_rate(5000, 1000, 1.0, 30) == pytest.approx(5.0, rel=0.03)


def test_shrink_rate_is_vectorized_and_lies_between_raw_and_prior():
    n, e = np.array([10.0, 40.0]), np.array([10.0, 40.0])  # raw rate = 1.0 for both
    out = shrink_rate(n, e, 3.0, 30)
    assert out.shape == (2,) and 1.0 < out[1] < out[0] < 3.0  # more exposure -> closer to raw


# ---------------- spec integrity ----------------
def test_six_attributes_each_have_features_and_valid_directions():
    assert ATTRIBUTES == ["ATT", "DEF", "WRE", "SUB", "STA", "PSY"]
    assert {f.attribute for f in FEATURES.values()} == set(ATTRIBUTES)
    assert all(f.direction in (1, -1) for f in FEATURES.values())


# ---------------- PIT counts ----------------
def test_first_bout_has_zero_history_and_second_sees_only_the_first(counts):
    first, second = row(counts, "Alpha One", "aaa1"), row(counts, "Alpha One", "ccc3")
    assert first["prior_bouts"] == 0 and first["prior_sig_landed"] == 0
    assert (second["prior_bouts"], second["prior_wins"], second["prior_sig_landed"], second["prior_stat_sec"]) == (1, 1, 28, 390)


def test_prior_columns_equal_shifted_cumulative_sums_for_every_fighter(counts):
    real = counts[~counts["is_current"]].sort_values(["fighter", "event_date", "bout_id"])
    for metric in METRICS:
        expected = real.groupby("fighter")[metric].cumsum() - real[metric]
        assert (real[f"prior_{metric}"].astype(float).values == pytest.approx(expected.astype(float).values)), metric


def test_one_current_row_per_fighter_carries_full_career(counts):
    cur = counts[counts["is_current"]]
    assert len(cur) == 6 and cur["fighter"].is_unique
    a = cur[cur["fighter"] == "Alpha One"].iloc[0]
    assert (a["prior_bouts"], a["prior_wins"], a["prior_sig_landed"], a["prior_sig_att"], a["prior_stat_sec"]) == (2, 2, 33, 48, 640)
    assert (a["prior_sub_wins"], a["prior_sub_att"], a["prior_ctrl_sec"], a["prior_kd"]) == (1, 2, 240, 2)


def test_round_one_and_late_round_exposure(counts):
    a = counts[(counts["fighter"] == "Alpha One") & counts["is_current"]].iloc[0]
    assert (a["prior_r1_landed"], a["prior_r1_sec"], a["prior_late_sec"]) == (25, 550, 0)


def test_bounce_back_counts_only_bouts_right_after_a_loss(counts):
    bravo = counts[(counts["fighter"] == "Bravo Two") & counts["is_current"]].iloc[0]
    assert (bravo["prior_after_loss"], bravo["prior_after_loss_win"]) == (1, 0)  # lost, then drew
    assert row(counts, "Bravo Two", "aaa1")["after_loss"] == 0


def test_opponent_strength_uses_opponent_record_at_fight_time(counts):
    a = counts[(counts["fighter"] == "Alpha One") & counts["is_current"]].iloc[0]
    assert a["prior_opp_n"] == 2
    assert a["prior_opp_wr_sum"] == pytest.approx(0.5 + 2 / 3)  # Bravo 0-0 then Charlie 1-0 (Laplace smoothed)


def test_loss_durability_is_fraction_of_scheduled_time(counts):
    bravo = counts[(counts["fighter"] == "Bravo Two") & counts["is_current"]].iloc[0]
    assert bravo["prior_loss_dur_n"] == 1 and bravo["prior_loss_dur_sum"] == pytest.approx(390 / 900)


# ---------------- priors + features ----------------
def test_priors_pool_only_bouts_up_to_the_cutoff(counts):
    pri = fit_priors(counts, cutoff=date(2024, 12, 31))  # only bouts aaa1 / bbb2 are eligible
    lw = pri[(pri["gender"] == "M") & (pri["weight_class"] == "Lightweight")].iloc[0]
    assert lw["sig_acc"] == pytest.approx(40 / 73)  # (28 + 12) landed / (40 + 33) attempted
    assert ((pri["gender"] == "*") & (pri["weight_class"] == "*")).any()  # global fallback row exists


def test_features_are_finite_bounded_and_fighter_without_stats_gets_the_prior(counts):
    pri = fit_priors(counts, cutoff=date(2026, 1, 1))
    feats = compute_features(counts, pri)
    assert set(FEATURES) <= set(feats.columns) and np.isfinite(feats[list(FEATURES)].to_numpy()).all()
    assert feats["sig_acc"].between(0, 1).all() and feats["sig_def"].between(0, 1).all()
    glob = pri[pri["gender"] == "*"].iloc[0]
    echo = feats[(counts["fighter"] == "Echo Five") & counts["is_current"]].iloc[0]  # NC bout, no stats at all
    assert echo["sig_acc"] == pytest.approx(glob["sig_acc"])


def test_thin_sample_is_pulled_toward_prior(counts):
    pri = fit_priors(counts, cutoff=date(2026, 1, 1))
    feats = compute_features(counts, pri)
    i = counts[(counts["fighter"] == "Alpha One") & counts["is_current"]].index[0]
    raw, prior = 33 / 48, pri[pri["gender"] == "*"].iloc[0]["sig_acc"]
    assert min(raw, prior) < feats.loc[i, "sig_acc"] < max(raw, prior)
