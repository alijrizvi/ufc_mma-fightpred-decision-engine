"""Slice 2 integration gate on the REAL snapshot (skipped unless UFC_RAW_DIR is set)."""
import os
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.engine.features import ATTRIBUTES, build_pit_counts
from src.engine.ingest import load_snapshot
from src.engine.medallion import UFCWarehouse
from src.engine.ratings import build_cards, build_training_frame, run_training

RAW_DIR = os.environ.get("UFC_RAW_DIR")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not RAW_DIR, reason="set UFC_RAW_DIR to run")]
SHOWCASE = ["Islam Makhachev", "Dustin Poirier", "Justin Gaethje", "Carlos Prates",
            "Ilia Topuria", "Charles Oliveira", "Max Holloway", "Arman Tsarukyan"]
AS_OF = date.today()


@pytest.fixture(scope="module")
def trained():
    warnings.simplefilter("ignore")
    wh = UFCWarehouse()
    wh.load_bronze(load_snapshot(Path(RAW_DIR)))
    wh.build_silver()
    wh.build_gold(as_of=AS_OF)
    counts = build_pit_counts(wh, as_of=AS_OF)
    version, priors, feats = run_training(counts, cutoff=date(2023, 12, 31))
    return wh, counts, version, priors, feats


def test_orientation_leaves_labels_balanced(trained):
    _, counts, _, _, feats = trained
    assert 0.47 < build_training_frame(counts, feats)["y"].mean() < 0.53


def test_out_of_time_model_beats_a_coin_flip(trained):
    metrics = trained[2].metrics
    assert metrics["n"] > 1000 and metrics["log_loss"] < 0.6931 and metrics["auc"] > 0.58


def test_all_showcase_fighters_get_valid_non_provisional_cards(trained):
    wh, counts, version, priors, feats = trained
    cur = counts[counts["is_current"]]
    current = feats.loc[cur.index].copy()
    current.insert(0, "fighter", cur["fighter"].to_numpy())
    info = wh.query("SELECT fighter, gender, latest_weight_class, status, is_provisional, ufc_bouts, last_fight_date "
                    "FROM gold.dim_fighter")
    cards = build_cards(current, info, version.model).set_index("fighter")
    for name in SHOWCASE:
        assert name in cards.index, name
        assert cards.loc[name, ATTRIBUTES].between(65, 99).all()
        assert not cards.loc[name, "is_provisional"]
    assert pd.api.types.is_integer_dtype(cards["ATT"])
