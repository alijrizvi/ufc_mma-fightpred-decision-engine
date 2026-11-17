import re
from typing import Any, Dict, List, Optional, Tuple

from src.agent.schemas import Claim
from src.engine.app_config import AUDIT_TOLERANCE_PCT

_EPS = 1e-9
EXACT_TOL_PCT = 0.1  # rounding slack for numbers quoted to one decimal (CI bounds, method shares)


def _pct(x: float) -> float:
    """Fractions (<=1) become percentage points; values above 1 are assumed to be percentages already."""
    return x * 100.0 if x <= 1.0 else x


class AnalyticalAuditor:
    """Receipt-style auditor verifying narrator claims against ground truth within tolerance."""

    def __init__(self, tolerance_pct: float = AUDIT_TOLERANCE_PCT) -> None:
        self.tolerance_pct = tolerance_pct

    def _tol(self, tolerance_pct: Optional[float]) -> float:
        return self.tolerance_pct if tolerance_pct is None else tolerance_pct

    def verify_claims(
        self,
        text_response: str,
        ground_truth: Dict[str, Any],
        tolerance_pct: Optional[float] = None,
    ) -> Tuple[bool, List[str]]:
        """Safety net: every percentage written in the prose must match some ground-truth number."""
        tol = self._tol(tolerance_pct)
        pct_matches = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", text_response)]

        win_targets = [
            round(_pct(ground_truth.get("win_rate_a", 0.0)), 1),
            round(_pct(ground_truth.get("win_rate_b", 0.0)), 1),
        ]
        # CI bounds and method shares are quoted to one decimal, so they get a tight window:
        # a loose one would let a fake win rate hide next to a nearby real number.
        exact_targets = [round(_pct(v), 1) for key in ("win_ci_a", "win_ci_b") for v in ground_truth.get(key) or []]
        exact_targets += [round(_pct(v), 1) for v in (ground_truth.get("method_breakdown") or {}).values()]

        violations = []
        for val in pct_matches:
            matched = any(abs(val - t) <= tol + _EPS for t in win_targets) or any(
                abs(val - t) <= EXACT_TOL_PCT + _EPS for t in exact_targets
            )
            if not matched and val not in [0.0, 100.0]:
                violations.append(f"Unverified numeric claim: {val}% does not match ground truth {win_targets}.")
        return len(violations) == 0, violations

    def verify_structured(
        self,
        claims: List[Claim],
        ground_truth: Dict[str, Any],
        tolerance_pct: Optional[float] = None,
    ) -> Tuple[bool, List[str]]:
        """Checks each structured claim (the itemized receipt) against ground truth.

        Returns:
            (is_valid, violations). No claims at all is itself a failure: unverifiable.
        """
        tol = self._tol(tolerance_pct)
        if not claims:
            return False, ["No verifiable claims were provided."]

        fighters = {ground_truth.get("fighter_a"): "a", ground_truth.get("fighter_b"): "b"}
        methods = ground_truth.get("method_breakdown") or {}
        violations: List[str] = []

        for c in claims:
            side = fighters.get(c.subject)
            if c.type == "fighter":
                if side is None:
                    violations.append(f"Unknown fighter '{c.subject}' is not in this matchup.")
            elif c.type == "win_prob":
                if side is None or c.value is None:
                    violations.append(f"Win probability for '{c.subject}' cannot be verified.")
                    continue
                exp = _pct(ground_truth[f"win_rate_{side}"])
                if abs(c.value - exp) > tol + _EPS:
                    violations.append(f"{c.subject} win probability {c.value}% does not match ground truth {exp:.1f}%.")
            elif c.type == "method_share":
                if c.subject not in methods or c.value is None:
                    violations.append(f"Unknown finish method '{c.subject}'.")
                    continue
                exp = _pct(methods[c.subject])
                if abs(c.value - exp) > tol + _EPS:
                    violations.append(f"{c.subject} share {c.value}% does not match ground truth {exp:.1f}%.")
            elif c.type == "ci":
                ci = ground_truth.get(f"win_ci_{side}") if side else None
                if not ci or c.lo is None or c.hi is None:
                    violations.append(f"Confidence interval for '{c.subject}' cannot be verified.")
                    continue
                exp_lo, exp_hi = _pct(ci[0]), _pct(ci[1])
                if abs(c.lo - exp_lo) > tol + _EPS or abs(c.hi - exp_hi) > tol + _EPS:
                    violations.append(
                        f"{c.subject} CI [{c.lo}, {c.hi}]% does not match ground truth [{exp_lo:.1f}, {exp_hi:.1f}]%."
                    )
            elif c.type == "provenance":
                if c.subject != "n_simulations" or c.value != ground_truth.get("total_simulations"):
                    violations.append(
                        f"Provenance claim {c.subject}={c.value} does not match "
                        f"n_simulations={ground_truth.get('total_simulations')}."
                    )

        return len(violations) == 0, violations
