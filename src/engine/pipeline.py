"""Weekly pipeline: ingest -> medallion -> features -> train -> gate -> publish."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.engine import ratings
from src.engine.features import build_pit_counts
from src.engine.ingest import load_snapshot
from src.engine.medallion import UFCWarehouse
from src.engine.ratings import (
    DEFAULT_TRAIN_START,
    ModelVersion,
    evaluate_gate,
    load_latest_version,
    run_training,
    save_version,
)

TRAIN_CUTOFF = date(2023, 12, 31)


def passes_gate(prev: Optional[ModelVersion], new: ModelVersion) -> bool:
    """True when the challenger may replace the champion."""
    return evaluate_gate(prev, new).promote


def build_cards(
    as_of: str, raw_dir: Path, cutoff: date = TRAIN_CUTOFF,
    train_start: date = DEFAULT_TRAIN_START,
) -> Tuple[pd.DataFrame, ModelVersion]:
    """Run everything up to (but not including) publishing. Clock is injected."""
    day = date.fromisoformat(as_of)
    wh = UFCWarehouse()
    wh.load_bronze(load_snapshot(raw_dir), ingested_at=datetime.combine(day, datetime.min.time()))
    wh.build_silver()
    wh.build_gold(as_of=day)
    counts = build_pit_counts(wh, as_of=day)
    version, _priors, feats = run_training(
        counts, cutoff, train_start=train_start, version_id=f"v{as_of}", created_at=as_of
    )
    now = counts["is_current"].astype(bool)
    current = pd.concat([counts.loc[now, ["fighter"]], feats.loc[now]], axis=1)
    fighters = wh.query(
        "SELECT fighter, gender, latest_weight_class, status, is_provisional, "
        "ufc_bouts, last_fight_date FROM gold.dim_fighter"
    )
    cards = ratings.build_cards(current, fighters, version.model)
    return cards.sort_values("fighter").reset_index(drop=True), version


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def run_weekly(as_of: str, raw_dir: Path, out_dir: Path, cutoff: date = TRAIN_CUTOFF,
               train_start: date = DEFAULT_TRAIN_START) -> bool:
    """Publish cards only if the gate passes. Returns True when published."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    registry = out / "registry"
    cards, version = build_cards(as_of, Path(raw_dir), cutoff, train_start)  # all heavy work first
    if not passes_gate(load_latest_version(registry), version):
        return False  # champion stays, nothing touched
    save_version(registry, version)
    _atomic_write(out / "cards.json",
                  cards.to_json(orient="records", date_format="iso", indent=2))
    meta = {"as_of": as_of, "version_id": version.version_id,
            "train_cutoff": version.train_cutoff, "metrics": version.metrics}
    _atomic_write(out / "metrics.json", json.dumps(meta, sort_keys=True, indent=2, default=float))
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", default="artifacts/cards")
    ap.add_argument("--as-of", default=date.today().isoformat())  # the ONLY clock read
    a = ap.parse_args()
    print("published" if run_weekly(a.as_of, Path(a.raw_dir), Path(a.out_dir)) else "held")
