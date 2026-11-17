"""Governed semantic layer: metric names -> fields on the published fighter cards.

A registry (ADR 024): the pipeline already computes every value, so the layer only
says what each metric is called, where it lives, and what range is believable.
Anything not in config/metrics.yml fails loudly instead of being guessed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH = ROOT / "config" / "metrics.yml"
DEFAULT_CARDS_PATH = ROOT / "artifacts" / "cards" / "cards.json"


class SemanticError(ValueError):
    """Base class (a ValueError, so older callers catching ValueError still work)."""


class UnknownMetricError(SemanticError):
    """The metric is not defined in metrics.yml."""


class UnknownFighterError(SemanticError):
    """The fighter has no published card."""


class MetricRangeError(SemanticError):
    """The card value is missing, non-numeric, or outside the declared range."""


class SemanticLayer:
    """Loads the metric registry and the published cards; serves governed values."""

    def __init__(self, metrics_path: Path | str, cards_path: Path | str) -> None:
        raw = yaml.safe_load(Path(metrics_path).read_text()) or {}
        self._metrics: dict[str, dict[str, Any]] = raw.get("metrics") or {}
        cards = json.loads(Path(cards_path).read_text())
        self._cards: dict[str, dict[str, Any]] = {c["fighter"]: c for c in cards}

    @classmethod
    def from_defaults(cls) -> SemanticLayer:
        return cls(DEFAULT_METRICS_PATH, DEFAULT_CARDS_PATH)

    def metric_names(self) -> list[str]:
        return list(self._metrics)

    def fighters(self) -> list[str]:
        return list(self._cards)

    def load_metric(self, metric_name: str, fighter_name: str) -> float:
        """Returns one governed metric value for a fighter.

        Raises:
            UnknownMetricError: Metric not in the registry (message lists valid names).
            UnknownFighterError: No card for this fighter.
            MetricRangeError: Value missing, non-numeric, or outside [min, max].
        """
        spec = self._metrics.get(metric_name)
        if spec is None:
            raise UnknownMetricError(f"Unknown metric '{metric_name}'. Valid metrics: {', '.join(self._metrics)}.")
        card = self._cards.get(fighter_name)
        if card is None:
            raise UnknownFighterError(f"No card for fighter '{fighter_name}'.")
        value = card.get(spec["field"])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricRangeError(f"{metric_name} for {fighter_name} is missing or not numeric: {value!r}.")
        lo, hi = spec.get("min"), spec.get("max")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise MetricRangeError(f"{metric_name}={value} for {fighter_name} is outside [{lo}, {hi}].")
        return float(value)
