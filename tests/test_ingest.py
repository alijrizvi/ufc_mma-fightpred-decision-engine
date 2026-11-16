"""Slice 1 (RED): ingestion boundary - schema contract, snapshot loading, fail-loud behavior."""
from pathlib import Path

import pytest

from src.engine.ingest import (
    EXPECTED_COLUMNS,
    RAW_FILES,
    SchemaDriftError,
    fetch_snapshot,
    load_snapshot,
    validate_schema,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ufcstats"


def test_load_snapshot_returns_all_four_sources():
    snap = load_snapshot(FIXTURES)
    assert set(snap) == set(RAW_FILES)
    assert len(snap["fight_results"]) == 5


def test_snapshot_is_loaded_verbatim():
    snap = load_snapshot(FIXTURES)
    assert snap["fight_results"]["EVENT"].iloc[0] == "UFC 300: Test A "  # trailing space kept: cleaning is Silver's job


def test_validate_schema_accepts_expected_columns():
    snap = load_snapshot(FIXTURES)
    for name, df in snap.items():
        validate_schema(name, df)


def test_validate_schema_fails_loudly_on_missing_column():
    snap = load_snapshot(FIXTURES)
    broken = snap["fight_results"].drop(columns=["METHOD"])
    with pytest.raises(SchemaDriftError, match="METHOD"):
        validate_schema("fight_results", broken)


def test_validate_schema_tolerates_extra_columns():
    snap = load_snapshot(FIXTURES)
    extra = snap["fight_results"].assign(NEW_UPSTREAM_COLUMN=1)
    validate_schema("fight_results", extra)


def test_validate_schema_rejects_empty_frame():
    snap = load_snapshot(FIXTURES)
    with pytest.raises(SchemaDriftError, match="empty"):
        validate_schema("fight_results", snap["fight_results"].iloc[0:0])


def test_expected_columns_cover_every_source():
    assert set(EXPECTED_COLUMNS) == set(RAW_FILES)


def test_fetch_snapshot_downloads_validates_and_writes(tmp_path):
    written = fetch_snapshot(tmp_path, base_url=FIXTURES.as_uri() + "/", pause_s=0)
    assert {p.name for p in written} == set(RAW_FILES.values())
    assert len(load_snapshot(tmp_path)["fighters"]) == 7


def test_fetch_snapshot_never_overwrites_good_data_with_bad(tmp_path):
    fetch_snapshot(tmp_path, base_url=FIXTURES.as_uri() + "/", pause_s=0)
    bad_dir = tmp_path / "bad_upstream"
    bad_dir.mkdir()
    for fname in RAW_FILES.values():
        (bad_dir / fname).write_text("WRONG,COLUMNS\n1,2\n")
    with pytest.raises(SchemaDriftError):
        fetch_snapshot(tmp_path, base_url=bad_dir.as_uri() + "/", retries=1, pause_s=0)
    assert len(load_snapshot(tmp_path)["fight_results"]) == 5  # original snapshot intact
