import pathlib

from src.engine.app_config import DEFAULT_SIMS, DEFAULT_SPEED, MAX_SIMS, SPEED_DELAYS


def test_sim_cap_is_200():
    assert MAX_SIMS == 200 and 0 < DEFAULT_SIMS <= MAX_SIMS


def test_medium_is_point_two_seconds():
    assert DEFAULT_SPEED == "Medium" and SPEED_DELAYS["Medium"] == 0.2


def test_speeds_get_faster_left_to_right():
    d = list(SPEED_DELAYS.values())
    assert d == sorted(d, reverse=True)


def test_app_uses_the_config():
    s = pathlib.Path("app.py").read_text()
    assert "MAX_SIMS" in s and "SPEED_DELAYS" in s and "max_value=500" not in s
