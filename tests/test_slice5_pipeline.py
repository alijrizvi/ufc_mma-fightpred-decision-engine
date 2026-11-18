"""Slice 5 (Red): --fetch, publish integrity gate, rating-shift report, empty-population percentile, cron, docker."""
import pathlib

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _cards(n=30, rating=80):
    rows = [{"fighter": f"F{i}", **{a: rating for a in ("ATT", "DEF", "WRE", "SUB", "STA", "PSY")}} for i in range(n)]
    return pd.DataFrame(rows)


def test_percentile_of_empty_population_is_neutral():
    from src.engine.ratings import percentile_of

    assert percentile_of(1.0, np.array([])) == 0.5


def test_validate_cards_accepts_good_cards():
    from src.engine.pipeline import validate_cards

    assert validate_cards(_cards()) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.iloc[0:0], "empty"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "duplicate"),
    (lambda d: d.assign(ATT=d["ATT"].astype(float).where(d.index != 3)), "missing"),
    (lambda d: d.assign(DEF=d["DEF"].where(d.index != 3, 120)), "range"),
])
def test_validate_cards_flags_problems(mutate, needle):
    from src.engine.pipeline import validate_cards

    problems = validate_cards(mutate(_cards()))
    assert problems and any(needle in p for p in problems)


def test_validate_cards_flags_big_roster_drop():
    from src.engine.pipeline import validate_cards

    assert any("shrank" in p for p in validate_cards(_cards(n=10), prev_count=100))
    assert validate_cards(_cards(n=90), prev_count=100) == []


def test_rating_shift_report():
    from src.engine.pipeline import rating_shift_report

    rep = rating_shift_report(_cards(rating=80), _cards(rating=95))
    assert rep["ATT"]["mean_abs_change"] == pytest.approx(15.0)
    assert rep["ATT"]["share_moved_over_10"] == pytest.approx(1.0)
    assert rating_shift_report(None, _cards()) == {}


def test_bad_cards_raise_and_publish_nothing(tmp_path, monkeypatch):
    from src.engine import pipeline

    monkeypatch.setattr(pipeline, "build_cards", lambda *a, **k: (_cards().iloc[0:0], object()))
    monkeypatch.setattr(pipeline, "load_latest_version", lambda r: None)
    monkeypatch.setattr(pipeline, "passes_gate", lambda *a, **k: True)
    with pytest.raises(pipeline.CardIntegrityError):
        pipeline.run_weekly("2026-11-16", tmp_path, tmp_path)
    assert not (tmp_path / "cards.json").exists()


def test_fetch_flag_downloads_before_running(tmp_path, monkeypatch):
    from src.engine import pipeline

    calls = []
    monkeypatch.setattr(pipeline, "run_weekly", lambda *a, **k: calls.append("run") or True)
    code = pipeline.main(["--raw-dir", str(tmp_path), "--as-of", "2026-11-16", "--fetch"],
                         fetch=lambda d: calls.append(("fetch", str(d))))
    assert code == 0 and calls == [("fetch", str(tmp_path)), "run"]


def test_no_fetch_flag_never_touches_network(tmp_path, monkeypatch):
    from src.engine import pipeline

    monkeypatch.setattr(pipeline, "run_weekly", lambda *a, **k: True)
    pipeline.main(["--raw-dir", str(tmp_path), "--as-of", "2026-11-16"], fetch=lambda d: pytest.fail("fetched"))


def test_weekly_workflow_is_scheduled_and_scoped():
    text = (ROOT / ".github" / "workflows" / "weekly.yml").read_text()
    wf = yaml.safe_load(text)
    on = wf.get("on") or wf.get(True)
    assert on["schedule"][0]["cron"] and "workflow_dispatch" in on
    assert wf["permissions"] == {"contents": "write"}
    assert "--fetch" in text and "artifacts/cards" in text and "pytest" in text


def test_docker_context_is_lean():
    ignore = (ROOT / ".dockerignore").read_text()
    for entry in (".git", "__pycache__", "data/raw"):
        assert entry in ignore
    assert (ROOT / "artifacts" / "cards" / "cards.json").exists()  # the image ships the published cards
