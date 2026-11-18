"""Slice 5 (Red): cards drive the simulator through ONE governed bridge (no legacy seeded table)."""
import inspect

import pytest

from src.engine.semantic import SemanticLayer, UnknownFighterError

R = {"ATT": 65, "DEF": 65, "WRE": 65, "SUB": 65, "STA": 65, "PSY": 65}


def _layer(tmp_path, cards):
    import json

    import yaml

    metrics = {"version": 2, "metrics": {k: {"field": k, "description": k, "min": 0, "max": 100} for k in R}}
    (tmp_path / "m.yml").write_text(yaml.safe_dump(metrics))
    (tmp_path / "c.json").write_text(json.dumps(cards))
    return SemanticLayer(tmp_path / "m.yml", tmp_path / "c.json")


def test_bounds_and_direction(tmp_path):
    from src.engine.profiles import profile_from_layer

    layer = _layer(tmp_path, [{"fighter": "Low", **R}, {"fighter": "High", **{k: 99 for k in R}}])
    lo, hi = profile_from_layer(layer, "Low"), profile_from_layer(layer, "High")
    for good_when_higher in ("punch_power", "arm_block_pct", "takedown_success_pct", "submission_conversion_pct"):
        assert getattr(hi, good_when_higher) > getattr(lo, good_when_higher)
    for good_when_lower in ("stamina_decay_rate", "durability_index"):
        assert getattr(hi, good_when_lower) < getattr(lo, good_when_lower)
    assert 0.0 < lo.arm_block_pct < hi.arm_block_pct < 0.95  # stays inside the combat engine's valid range


def test_unknown_fighter_raises(tmp_path):
    from src.engine.profiles import profile_from_layer

    with pytest.raises(UnknownFighterError):
        profile_from_layer(_layer(tmp_path, [{"fighter": "A", **R}]), "Nobody")


def test_stance_label_is_deterministic(tmp_path):
    from src.engine.profiles import profile_from_layer, stance_label

    layer = _layer(tmp_path, [{"fighter": "A", **R}])
    assert stance_label(profile_from_layer(layer, "A")) == stance_label(profile_from_layer(layer, "A"))


def test_simulation_uses_cards_not_legacy_table(tmp_path, monkeypatch):
    from src.agent import tools

    cards = [{"fighter": "Strong", **{k: 99 for k in R}}, {"fighter": "Weak", **R}]
    monkeypatch.setattr(tools, "get_semantic_layer", lambda: _layer(tmp_path, cards))
    out = tools.run_stochastic_simulation("Strong", "Weak", num_simulations=200)
    assert out["win_rate_a"] > out["win_rate_b"]
    assert "OctagonDataEngine" not in inspect.getsource(tools)


def test_showcase_roster_exists_in_published_cards():
    from src.engine.app_config import SHOWCASE_ROSTER

    layer = SemanticLayer.from_defaults()
    assert len(SHOWCASE_ROSTER) == 8 and set(SHOWCASE_ROSTER) <= set(layer.fighters())
