import hashlib
import pathlib

import pytest

from src.engine import pipeline


def _hash(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(pathlib.Path(d).glob("*")) if p.is_file()}

def test_idempotent(tmp_path, fixture_raw):
    pipeline.run_weekly(as_of="2026-11-16", raw_dir=fixture_raw, out_dir=tmp_path)
    first = _hash(tmp_path)
    pipeline.run_weekly(as_of="2026-11-16", raw_dir=fixture_raw, out_dir=tmp_path)
    assert first == _hash(tmp_path)

def test_gate_fail_publishes_nothing_new(tmp_path, fixture_raw, monkeypatch):
    pipeline.run_weekly(as_of="2026-11-09", raw_dir=fixture_raw, out_dir=tmp_path)
    before = _hash(tmp_path)
    monkeypatch.setattr(pipeline, "passes_gate", lambda *a, **k: False)
    pipeline.run_weekly(as_of="2026-11-16", raw_dir=fixture_raw, out_dir=tmp_path)
    assert before == _hash(tmp_path)

def test_crash_leaves_last_week_intact(tmp_path, fixture_raw, monkeypatch):
    pipeline.run_weekly(as_of="2026-11-09", raw_dir=fixture_raw, out_dir=tmp_path)
    before = _hash(tmp_path)
    def boom(*a, **k): raise RuntimeError("mid-run crash")
    monkeypatch.setattr(pipeline, "build_cards", boom)
    with pytest.raises(RuntimeError):
        pipeline.run_weekly(as_of="2026-11-16", raw_dir=fixture_raw, out_dir=tmp_path)
    assert before == _hash(tmp_path)
