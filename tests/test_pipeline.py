import hashlib
import pathlib

import pytest

from src.engine import pipeline


def _hash(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(pathlib.Path(d).glob("*")) if p.is_file()}

def test_idempotent(tmp_path, fixture_raw, fx_window):
    pipeline.run_weekly(as_of="2026-11-16", raw_dir=fixture_raw, out_dir=tmp_path, **fx_window)
    first = _hash(tmp_path)
    pipeline.run_weekly(as_of="2026-11-16", raw_dir=fixture_raw, out_dir=tmp_path, **fx_window)
    assert first == _hash(tmp_path)

def test_gate_fail_publishes_nothing_new(tmp_path, fixture_raw, fx_window, monkeypatch):
    pipeline.run_weekly(as_of="2026-11-09", raw_dir=fixture_raw, out_dir=tmp_path, **fx_window)
    before = _hash(tmp_path)
    monkeypatch.setattr(pipeline, "passes_gate", lambda *a, **k: False)
    pipeline.run_weekly(as_of="2026-11-16", raw_dir=fixture_raw, out_dir=tmp_path, **fx_window)
    assert before == _hash(tmp_path)

def test_crash_leaves_last_week_intact(tmp_path, fixture_raw, fx_window, monkeypatch):
    pipeline.run_weekly(as_of="2026-11-09", raw_dir=fixture_raw, out_dir=tmp_path, **fx_window)
    before = _hash(tmp_path)
    def boom(*a, **k): raise RuntimeError("mid-run crash")
    monkeypatch.setattr(pipeline, "build_cards", boom)
    with pytest.raises(RuntimeError):
        pipeline.run_weekly(as_of="2026-11-16", raw_dir=fixture_raw, out_dir=tmp_path, **fx_window)
    assert before == _hash(tmp_path)


@pytest.fixture(autouse=True)
def _fast(fast_training):
    pass
