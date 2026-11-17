"""Slice 2 (RED): out-of-time Elastic-Net, promotion gate, model versions, and FIFA-style 65-99 cards."""
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine.features import ATTRIBUTES, FEATURES, build_pit_counts, compute_features, fit_priors
from src.engine.ingest import load_snapshot
from src.engine.medallion import UFCWarehouse
from src.engine.ratings import (
    FittedModel,
    ModelVersion,
    attribute_weights,
    build_cards,
    build_training_frame,
    evaluate,
    evaluate_gate,
    fit_elastic_net,
    load_latest_version,
    orientation_flip,
    percentile_of,
    save_version,
    split_out_of_time,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ufcstats"
FEATURE_NAMES = list(FEATURES)


def fixture_counts():
    wh = UFCWarehouse()
    wh.load_bronze(load_snapshot(FIXTURES), ingested_at=datetime(2026, 11, 15))
    wh.build_silver()
    wh.build_gold(as_of=date(2026, 1, 1))
    return build_pit_counts(wh, as_of=date(2026, 1, 1))


def synthetic_frame(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    beta = np.array([1.5, -1.0, 0.0, 0.0, 0.0, 0.5])
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta))).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["y"] = y
    df["event_date"] = pd.date_range("2015-01-01", periods=n, freq="D")
    return df, [f"f{i}" for i in range(6)]


def make_model(coefs, scale=1.0):
    return FittedModel(feature_names=list(coefs), coef=dict(coefs), scale={k: scale for k in coefs}, C=0.1, l1_ratio=0.5)


# ---------------- orientation (label-leak guard) ----------------
def test_orientation_is_deterministic_and_balanced():
    flips = [orientation_flip(f"bout{i:05d}") for i in range(2000)]
    assert flips == [orientation_flip(f"bout{i:05d}") for i in range(2000)]
    assert 0.45 < np.mean(flips) < 0.55


def test_training_frame_labels_follow_the_oriented_left_fighter():
    counts = fixture_counts()
    feats = compute_features(counts, fit_priors(counts, cutoff=date(2026, 1, 1)))
    frame = build_training_frame(counts, feats)
    assert len(frame) == 3 and set(frame["bout_id"]) == {"aaa1", "bbb2", "ccc3"}  # NC / draws excluded
    real = counts[~counts["is_current"]]
    for _, r in frame.iterrows():
        left = real[(real["bout_id"] == r["bout_id"]) & (real["fighter"] == r["left_fighter"])].iloc[0]
        assert r["y"] == int(left["result"] == "WIN")
    first = frame.iloc[0]
    li = counts[(counts["bout_id"] == first["bout_id"]) & (counts["fighter"] == first["left_fighter"])].index[0]
    ri = counts[(counts["bout_id"] == first["bout_id"]) & (counts["fighter"] == first["right_fighter"])].index[0]
    assert first["sig_acc"] == pytest.approx(feats.loc[li, "sig_acc"] - feats.loc[ri, "sig_acc"])


# ---------------- out-of-time split ----------------
def test_split_out_of_time_never_leaks_future_into_training():
    df, _ = synthetic_frame(200)
    train, valid = split_out_of_time(df, cutoff=date(2015, 4, 1))
    assert train["event_date"].max() <= pd.Timestamp("2015-04-01") < valid["event_date"].min()
    assert len(train) + len(valid) == 200


# ---------------- Elastic-Net ----------------
def test_elastic_net_recovers_signal_and_zeroes_noise():
    df, cols = synthetic_frame()
    model = fit_elastic_net(df, cols)
    assert model.coef["f0"] > 0.5 and model.coef["f1"] < -0.3 and model.coef["f5"] > 0.1
    assert max(abs(model.coef[c]) for c in ("f2", "f3", "f4")) < 0.15
    assert model.C > 0 and 0 < model.l1_ratio < 1


def test_model_is_antisymmetric_so_swapping_fighters_flips_probability():
    df, cols = synthetic_frame()
    model = fit_elastic_net(df, cols)
    X = df[cols].head(50)
    assert (model.predict_proba(X) + model.predict_proba(-X)).to_numpy() == pytest.approx(1.0)


def test_evaluate_reports_honest_metrics():
    df, cols = synthetic_frame()
    train, valid = split_out_of_time(df, cutoff=date(2022, 1, 1))
    metrics = evaluate(fit_elastic_net(train, cols), valid)
    assert set(metrics) >= {"log_loss", "brier", "auc", "accuracy", "n"}
    assert metrics["auc"] > 0.7 and metrics["log_loss"] < 0.69 and metrics["n"] == len(valid)


# ---------------- promotion gate (ADR 017) ----------------
def version(coefs, log_loss=0.65, vid="v1"):
    return ModelVersion(version_id=vid, created_at="2026-11-15T08:00:00", train_cutoff="2023-12-31",
                        metrics={"log_loss": log_loss}, model=make_model(coefs))


BASE = {"a": 1.0, "b": -0.8, "c": 0.6, "d": 0.4, "e": 0.3, "f": 0.01}


def test_gate_promotes_first_ever_model():
    assert evaluate_gate(None, version(BASE)).promote


def test_gate_promotes_small_drift_within_tolerance():
    new = dict(BASE, a=1.1, c=0.62)
    assert evaluate_gate(version(BASE), version(new, log_loss=0.651)).promote


def test_gate_holds_when_log_loss_gets_worse():
    d = evaluate_gate(version(BASE), version(BASE, log_loss=0.70))
    assert not d.promote and any("log_loss" in r for r in d.reasons)


def test_gate_holds_on_sign_flip_in_a_top_coefficient():
    d = evaluate_gate(version(BASE), version(dict(BASE, b=+0.5)))
    assert not d.promote and any("sign" in r for r in d.reasons)


def test_gate_holds_when_more_than_one_top_coefficient_drifts_over_20_percent():
    d = evaluate_gate(version(BASE), version(dict(BASE, a=1.5, c=0.9)))
    assert not d.promote and any("drift" in r for r in d.reasons)


def test_gate_tolerates_a_single_drifting_coefficient():
    assert evaluate_gate(version(BASE), version(dict(BASE, a=1.5))).promote


# ---------------- versions ----------------
def test_versions_roundtrip_and_latest_wins(tmp_path):
    save_version(tmp_path, version(BASE, vid="2026-11-08T080000"))
    save_version(tmp_path, version(dict(BASE, a=2.0), vid="2026-11-15T080000"))
    latest = load_latest_version(tmp_path)
    assert latest.version_id == "2026-11-15T080000" and latest.model.coef["a"] == 2.0
    assert load_latest_version(tmp_path / "missing") is None


# ---------------- attribute weights ----------------
def full_coefs(**over):
    coefs = {f: 0.2 * spec.direction for f, spec in FEATURES.items()}
    coefs.update(over)
    return make_model(coefs)


def test_weights_follow_domain_direction_and_report_contradictions():
    w = attribute_weights(full_coefs(slpm=-0.5, avg_win_sec=-0.4))
    assert w["ATT"].weights["slpm"] == 0 and "slpm" in w["ATT"].contradictions
    assert w["ATT"].weights["avg_win_sec"] == pytest.approx(0.4)  # negative coef is the *expected* sign here
    assert not w["ATT"].fallback


def test_attribute_with_all_weights_zeroed_falls_back_to_equal_weights_and_says_so():
    zero_def = {f: 0.0 for f, s in FEATURES.items() if s.attribute == "DEF"}
    w = attribute_weights(full_coefs(**zero_def))
    assert w["DEF"].fallback and len(set(w["DEF"].weights.values())) == 1


# ---------------- percentile + cards ----------------
def test_percentile_of_handles_ties_and_extremes():
    pop = np.array([1.0, 2.0, 3.0, 4.0])
    assert percentile_of(0.0, pop) == 0.0 and percentile_of(5.0, pop) == 1.0
    assert percentile_of(2.0, pop) == pytest.approx(0.375)


def card_inputs(n=40, seed=3):
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    feats.insert(0, "fighter", [f"F{i:02d}" for i in range(n)])
    info = pd.DataFrame({
        "fighter": feats["fighter"], "gender": "M", "latest_weight_class": "Lightweight",
        "status": "ACTIVE", "is_provisional": False, "ufc_bouts": 12, "last_fight_date": pd.Timestamp("2026-06-01"),
    })
    return feats, info


def test_cards_are_integers_within_65_to_99_and_ordered_by_composite():
    feats, info = card_inputs()
    cards = build_cards(feats, info, full_coefs())
    assert list(cards.columns[:8]) == ["fighter", "division", *ATTRIBUTES]
    assert cards[ATTRIBUTES].apply(lambda s: s.between(65, 99).all()).all()
    assert cards[ATTRIBUTES].dtypes.apply(lambda d: np.issubdtype(d, np.integer)).all()
    best = feats.loc[feats["slpm"].idxmax(), "fighter"]  # slpm is a positive ATT feature
    assert cards.set_index("fighter").loc[best, "ATT"] >= cards["ATT"].median()


def test_inactive_fighters_do_not_shift_other_fighters_ratings():
    feats, info = card_inputs()
    base = build_cards(feats, info, full_coefs()).set_index("fighter")
    extra = feats.iloc[[0]].copy()
    extra["fighter"] = "RETIRED"
    extra.iloc[0, 1:] = 50.0  # absurd outlier
    feats2 = pd.concat([feats, extra], ignore_index=True)
    info2 = pd.concat([info, info.iloc[[0]].assign(fighter="RETIRED", status="INACTIVE")], ignore_index=True)
    again = build_cards(feats2, info2, full_coefs()).set_index("fighter")
    assert (again.loc[base.index, ATTRIBUTES] == base[ATTRIBUTES]).all().all()


def test_provisional_fighters_get_a_flagged_card_but_stay_out_of_the_population():
    feats, info = card_inputs()
    info.loc[0, "is_provisional"] = True
    cards = build_cards(feats, info, full_coefs()).set_index("fighter")
    assert bool(cards.loc["F00", "is_provisional"]) and cards.loc["F00", ATTRIBUTES].between(65, 99).all()


def test_small_divisions_are_scaled_against_the_wider_gender_population():
    feats, info = card_inputs(n=40)
    info.loc[:2, "latest_weight_class"] = "Flyweight"  # only 3 fighters -> below min population
    cards = build_cards(feats, info, full_coefs(), min_population=15)
    assert cards[ATTRIBUTES].nunique().min() > 3  # not collapsed into 3-person ranks
