"""Bridge: published rating cards (65-99) -> the combat-simulation profile (ADR 026).

One governed adapter, so the agent's ground truth and the Streamlit simulator read the same
numbers. Every rating goes through SemanticLayer.load_metric (name check + range check).
v1 is a monotone linear proxy: each parameter slides between a weak value (rating 65) and a
strong value (rating 99). The endpoints are modelling assumptions, not fitted estimates.
"""
from __future__ import annotations

from src.engine.semantic import SemanticLayer
from src.engine.stats import FighterCombatProfile

RATING_MIN, RATING_MAX = 65.0, 99.0

# parameter -> (card attribute, value at rating 65, value at rating 99)
PROFILE_MAP = {
    "punch_power": ("ATT", 14.0, 32.0),
    "elbow_power": ("ATT", 14.0, 28.0),
    "leg_kick_power": ("ATT", 9.0, 24.0),
    "strike_hit_pct": ("ATT", 0.40, 0.62),
    "arm_block_pct": ("DEF", 0.22, 0.50),
    "leg_block_pct": ("DEF", 0.18, 0.42),
    "durability_index": ("PSY", 1.15, 0.85),  # composure: lower multiplier = takes less damage
    "takedown_def_pct": ("WRE", 0.50, 0.92),
    "takedown_success_pct": ("WRE", 0.20, 0.65),
    "gnp_avg_damage": ("WRE", 9.0, 16.0),
    "submission_conversion_pct": ("SUB", 0.04, 0.35),
    "stamina_decay_rate": ("STA", 0.048, 0.024),  # better cardio = slower decay
}


def _lerp(rating: float, weak: float, strong: float) -> float:
    t = min(1.0, max(0.0, (rating - RATING_MIN) / (RATING_MAX - RATING_MIN)))
    return round(weak + t * (strong - weak), 4)


def profile_from_layer(layer: SemanticLayer, fighter_name: str) -> FighterCombatProfile:
    """Builds a simulation profile from a fighter's governed card ratings.

    Raises:
        UnknownFighterError: No card for the fighter.
        MetricRangeError: A rating is missing, non-numeric, or out of range.
    """
    ratings = {attr: layer.load_metric(attr, fighter_name) for attr in {a for a, _, _ in PROFILE_MAP.values()}}
    params = {p: _lerp(ratings[attr], weak, strong) for p, (attr, weak, strong) in PROFILE_MAP.items()}
    return FighterCombatProfile(name=fighter_name, **params)


def stance_label(profile: FighterCombatProfile) -> str:
    """Display archetype derived from the profile (same rule the legacy view used)."""
    if profile.takedown_success_pct >= 0.50:
        return "Sambo / Grappling"
    if profile.leg_kick_power >= 18.0:
        return "Muay Thai / Striker"
    return "Orthodox Boxer-Puncher"
