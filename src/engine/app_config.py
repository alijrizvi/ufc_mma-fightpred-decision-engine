"""UI limits for live playback and Monte Carlo (one place, so tests can pin them)."""
from __future__ import annotations

MAX_SIMS = 200      # latency budget: a live replay must not keep the user waiting
MIN_SIMS = 50
DEFAULT_SIMS = 100
SIM_STEP = 50

# seconds between animation frames
SPEED_DELAYS = {"Slow": 0.4, "Medium": 0.2, "Fast": 0.1, "Turbo": 0.03}
DEFAULT_SPEED = "Medium"

# One audit tolerance for the whole system (percentage points); auditor, copilot and UI read this.
AUDIT_TOLERANCE_PCT = 1.5

# The 8 live lightweights shown in the simulator (chosen by interest, not by a weight-class rule).
SHOWCASE_ROSTER = (
    "Islam Makhachev", "Ilia Topuria", "Arman Tsarukyan", "Charles Oliveira",
    "Max Holloway", "Dustin Poirier", "Justin Gaethje", "Carlos Prates",
)
