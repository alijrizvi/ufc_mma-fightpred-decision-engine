"""Plain-Python analytical tools (the single source of truth).

`mcp_server.py` only wraps these with FastMCP; `harness.py` calls them directly.
"""

from functools import lru_cache
from typing import Any, Dict

from src.agent.schemas import MetricQueryResponse, SimulationResponse
from src.engine.etl import OctagonDataEngine
from src.engine.intervals import wilson_interval
from src.engine.monte_carlo import MonteCarloRiskEngine
from src.engine.semantic import SemanticLayer


def _seeded_db() -> OctagonDataEngine:
    db_engine = OctagonDataEngine()
    db_engine.seed_historical_lightweight_dataset()
    return db_engine


@lru_cache(maxsize=1)
def get_semantic_layer() -> SemanticLayer:
    """The governed registry over the published cards (loaded once)."""
    return SemanticLayer.from_defaults()


def query_semantic_metric(fighter_name: str, metric_name: str) -> Dict[str, Any]:
    """Reads one governed metric for a fighter from the published cards.

    Args:
        fighter_name: Fighter with a published card.
        metric_name: Metric defined in config/metrics.yml (e.g. "ATT").

    Returns:
        A MetricQueryResponse as a dict.

    Raises:
        UnknownMetricError: Metric not in the registry (a ValueError).
        UnknownFighterError: No card for the fighter (a ValueError).
        MetricRangeError: Card value missing or out of range (a ValueError).
    """
    value = round(get_semantic_layer().load_metric(metric_name, fighter_name), 4)
    return MetricQueryResponse(fighter_name=fighter_name, metric_name=metric_name, value=value).model_dump()


def run_stochastic_simulation(fighter_a: str, fighter_b: str, num_simulations: int = 1000) -> Dict[str, Any]:
    """Runs N Monte Carlo bouts and returns win-rate distributions.

    Raises:
        ValueError: If either fighter is not in the DuckDB profile table.
    """
    db_engine = _seeded_db()
    prof_a = db_engine.load_fighter_profile(fighter_a)
    prof_b = db_engine.load_fighter_profile(fighter_b)

    summary = MonteCarloRiskEngine(num_simulations=num_simulations).run_bout_simulation(prof_a, prof_b)
    wins_a = round(summary.win_rate_a * num_simulations)
    wins_b = round(summary.win_rate_b * num_simulations)
    return SimulationResponse(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        total_simulations=num_simulations,
        win_rate_a=summary.win_rate_a,
        win_rate_b=summary.win_rate_b,
        draw_rate=summary.draw_rate,
        avg_finish_interval=summary.avg_finish_interval,
        method_breakdown=summary.method_breakdown,
        win_ci_a=wilson_interval(wins_a, num_simulations),
        win_ci_b=wilson_interval(wins_b, num_simulations),
    ).model_dump()
