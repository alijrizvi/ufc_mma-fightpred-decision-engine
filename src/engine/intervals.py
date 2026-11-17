"""Confidence intervals for simulation proportions."""
from __future__ import annotations

import math

Z_95 = 1.96


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Unlike the plain normal ("Wald") interval, it stays inside [0, 1] and behaves
    well for small n or proportions near 0 or 1, which matters at N=200 sims.

    Args:
        successes: Number of wins observed (0 <= successes <= n).
        n: Number of simulations (> 0).
        z: Z-score for the confidence level (1.96 = 95%).

    Returns:
        (lower, upper) as fractions in [0, 1].

    Raises:
        ValueError: If n <= 0 or successes is outside [0, n].
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n.")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)
