import pathlib

from src.engine.card_view import render_card_html

CARD = {"fighter": "Ilia Topuria", "division": "Lightweight", "ATT": 95, "DEF": 88,
        "WRE": 80, "SUB": 70, "STA": 91, "PSY": 85, "status": "ACTIVE", "is_provisional": False}


def test_six_stats_in_two_columns():
    html = render_card_html(CARD)
    assert all(a in html for a in ("ATT", "DEF", "WRE", "SUB", "STA", "PSY"))
    assert html.count("flex:1") == 2 and ">95<" in html


def test_badges():
    assert "STALE" in render_card_html({**CARD, "status": "STALE"})
    assert "INACTIVE" in render_card_html({**CARD, "status": "INACTIVE"})
    assert "STALE" not in render_card_html(CARD)


def test_missing_card_and_escaping():
    assert "No rating card" in render_card_html(None, name="Nobody")
    assert "<script>" not in render_card_html(None, name="<script>")


def test_app_is_wired():
    assert "render_card_html" in pathlib.Path("app.py").read_text()
