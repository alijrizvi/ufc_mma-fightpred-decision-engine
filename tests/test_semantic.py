"""Slice 4 (Red): governed metric registry over the published cards. No attribute guessing."""
import json
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

CARDS = [
    {"fighter": "Ali Test", "ATT": 70, "DEF": 60, "ufc_bouts": 6},
    {"fighter": "Bo Example", "ATT": 58, "DEF": 72, "ufc_bouts": 9},
]
REGISTRY = {
    "version": 2,
    "metrics": {
        "ATT": {"field": "ATT", "description": "Attack rating", "min": 0, "max": 100},
        "DEF": {"field": "DEF", "description": "Defense rating", "min": 0, "max": 100},
        "ufc_bouts": {"field": "ufc_bouts", "description": "UFC bouts on record", "min": 0},
    },
}


@pytest.fixture
def layer_files(tmp_path):
    metrics, cards = tmp_path / "metrics.yml", tmp_path / "cards.json"
    metrics.write_text(yaml.safe_dump(REGISTRY))
    cards.write_text(json.dumps(CARDS))
    return metrics, cards


def test_load_metric_reads_value_from_card(layer_files):
    from src.engine.semantic import SemanticLayer

    layer = SemanticLayer(*layer_files)
    assert layer.load_metric("ATT", "Ali Test") == 70.0
    assert set(layer.metric_names()) == {"ATT", "DEF", "ufc_bouts"}


def test_unknown_metric_raises_and_lists_valid_names(layer_files):
    from src.engine.semantic import SemanticLayer, UnknownMetricError

    assert issubclass(UnknownMetricError, ValueError)  # old callers catching ValueError still work
    with pytest.raises(UnknownMetricError, match="ATT"):
        SemanticLayer(*layer_files).load_metric("punch_power", "Ali Test")


def test_unknown_fighter_raises(layer_files):
    from src.engine.semantic import SemanticLayer, UnknownFighterError

    assert issubclass(UnknownFighterError, ValueError)
    with pytest.raises(UnknownFighterError, match="Nobody"):
        SemanticLayer(*layer_files).load_metric("ATT", "Nobody")


def test_value_outside_declared_range_is_rejected(layer_files):
    from src.engine.semantic import MetricRangeError, SemanticLayer

    metrics, cards = layer_files
    cards.write_text(json.dumps([{"fighter": "Ali Test", "ATT": 140, "DEF": 60, "ufc_bouts": 6}]))
    with pytest.raises(MetricRangeError):
        SemanticLayer(metrics, cards).load_metric("ATT", "Ali Test")


def test_real_metrics_yml_is_the_cards_registry():
    raw = yaml.safe_load((ROOT / "config" / "metrics.yml").read_text())
    assert "cubes" not in raw, "stale interval_telemetry measures must be retired"
    cards = json.loads((ROOT / "artifacts" / "cards" / "cards.json").read_text())
    assert raw["metrics"], "registry must not be empty"
    for name, spec in raw["metrics"].items():
        assert spec["field"] in cards[0], f"{name}: field missing from cards.json"
        assert spec.get("description"), f"{name}: description required"


def test_every_real_card_satisfies_every_real_metric():
    from src.engine.semantic import SemanticLayer

    layer = SemanticLayer.from_defaults()
    cards = json.loads((ROOT / "artifacts" / "cards" / "cards.json").read_text())
    for card in cards:
        for name in layer.metric_names():
            layer.load_metric(name, card["fighter"])  # must not raise


def test_tool_uses_the_loader(layer_files, monkeypatch):
    from src.agent import tools
    from src.engine.semantic import SemanticLayer, UnknownMetricError

    monkeypatch.setattr(tools, "get_semantic_layer", lambda: SemanticLayer(*layer_files))
    assert tools.query_semantic_metric("Ali Test", "ATT")["value"] == 70.0
    with pytest.raises(UnknownMetricError):
        tools.query_semantic_metric("Ali Test", "punch_power")


def test_tool_source_has_no_attribute_guessing():
    import inspect

    from src.agent import tools

    source = inspect.getsource(tools.query_semantic_metric)
    assert "getattr" not in source and "hasattr" not in source
