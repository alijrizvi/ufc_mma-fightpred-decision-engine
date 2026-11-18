"""Weekly pipeline: ingest -> medallion -> features -> train -> gate -> publish."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.engine import ratings
from src.engine.features import ATTRIBUTES, build_pit_counts
from src.engine.ingest import fetch_snapshot, load_snapshot
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


class CardIntegrityError(ValueError):
    """New cards failed the publish checks; nothing was written."""


MAX_ROSTER_SHRINK = 0.20  # hold if the roster loses more than 20% of last week's fighters
RATING_LO, RATING_HI = 65, 99


def validate_cards(cards: pd.DataFrame, prev_count: Optional[int] = None) -> List[str]:
    """Publish gate on the cards themselves (the model gate only checks model metrics)."""
    if cards.empty:
        return ["cards are empty"]
    problems: List[str] = []
    if cards["fighter"].duplicated().any():
        problems.append("duplicate fighters")
    attrs = list(ATTRIBUTES)
    if cards[attrs].isna().any().any():
        problems.append("missing ratings (NaN)")
    elif ((cards[attrs] < RATING_LO) | (cards[attrs] > RATING_HI)).any().any():
        problems.append(f"rating out of range [{RATING_LO}, {RATING_HI}]")
    if prev_count and len(cards) < prev_count * (1 - MAX_ROSTER_SHRINK):
        problems.append(f"roster shrank from {prev_count} to {len(cards)} fighters")
    return problems


def rating_shift_report(prev: Optional[pd.DataFrame], new: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per-attribute week-over-week movement for fighters on both cards.

    Ratings are percentiles, so their overall distribution barely changes (a KS test would
    always look calm). Movement of the SAME fighter is the informative drift signal.
    """
    if prev is None or prev.empty:
        return {}
    both = prev.merge(new, on="fighter", suffixes=("_prev", "_new"))
    if both.empty:
        return {}
    report = {}
    for a in ATTRIBUTES:
        delta = (both[f"{a}_new"] - both[f"{a}_prev"]).abs()
        report[a] = {"mean_abs_change": round(float(delta.mean()), 4),
                     "share_moved_over_10": round(float((delta > 10).mean()), 4)}
    # Empty means nothing moved (keeps same-day re-runs byte-identical).
    return report if any(v["mean_abs_change"] for v in report.values()) else {}


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
    prev_cards = pd.read_json(out / "cards.json") if (out / "cards.json").exists() else None
    problems = validate_cards(cards, None if prev_cards is None else len(prev_cards))
    if problems:
        raise CardIntegrityError("; ".join(problems))  # loud: the weekly job should fail, not silently hold
    save_version(registry, version)
    _atomic_write(out / "cards.json",
                  cards.to_json(orient="records", date_format="iso", indent=2))
    meta = {"as_of": as_of, "version_id": version.version_id,
            "train_cutoff": version.train_cutoff, "metrics": version.metrics,
            "rating_shift": rating_shift_report(prev_cards, cards)}
    _atomic_write(out / "metrics.json", json.dumps(meta, sort_keys=True, indent=2, default=float))
    return True


def main(argv: Optional[List[str]] = None, fetch: Callable[..., object] = fetch_snapshot) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", default="artifacts/cards")
    ap.add_argument("--as-of", default=date.today().isoformat())  # the ONLY clock read
    ap.add_argument("--fetch", action="store_true", help="download fresh CSVs (validated, then swapped in) first")
    a = ap.parse_args(argv)
    if a.fetch:
        fetch(a.raw_dir)
    print("published" if run_weekly(a.as_of, Path(a.raw_dir), Path(a.out_dir)) else "held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
