from streamlit.testing.v1 import AppTest

from src.engine import pipeline

FIGHTERS = ["Makhachev", "Poirier", "Gaethje", "Prates",
            "Topuria", "Oliveira", "Holloway", "Tsarukyan"]

def test_cards_render(tmp_path, fixture_raw, monkeypatch):
    pipeline.run_weekly(as_of="2026-11-16", raw_dir=fixture_raw, out_dir=tmp_path)
    monkeypatch.setenv("CARDS_DIR", str(tmp_path))
    at = AppTest.from_file("app.py").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert all(f in text for f in FIGHTERS)
