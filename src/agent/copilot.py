"""Decision Copilot: the narration layer on top of the AgentHarness."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from anthropic import Anthropic

from src.agent.auditor import AnalyticalAuditor
from src.agent.harness import REFUSAL_PATTERN, AgentHarness
from src.agent.schemas import Claim, CopilotAnalysisResult, Narration
from src.engine.app_config import AUDIT_TOLERANCE_PCT

DEFAULT_MODEL = "claude-haiku-5-5"  # override with the ANTHROPIC_MODEL env var


class DecisionCopilot:
    REFUSAL_PATTERN = REFUSAL_PATTERN  # kept for backward compatibility

    def __init__(self, tolerance_pct: float = AUDIT_TOLERANCE_PCT, use_mock: bool = False, max_attempts: int = 2) -> None:
        self.use_mock = use_mock
        self.tolerance_pct = tolerance_pct
        self.max_attempts = max_attempts
        self.auditor = AnalyticalAuditor(tolerance_pct=tolerance_pct)
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.model_name = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.client = Anthropic(api_key=self.api_key) if (self.api_key and not use_mock) else None

    @staticmethod
    def _mock_narration(gt: dict[str, Any], fighter_a: str, fighter_b: str, attack: str | None) -> Narration:
        """Deterministic stand-in narrator. `attack` injects one specific lie so evals can prove the auditor catches it."""
        p_a = round(gt["win_rate_a"] * 100, 1)
        p_b = round(gt["win_rate_b"] * 100, 1)
        n_sims = gt.get("total_simulations", 1000)
        # Always 15 points from the truth, so a fake can never hide inside the audit tolerance.
        fake = round(p_a + 15.0 if p_a <= 50.0 else p_a - 15.0, 1)
        ci = gt.get("win_ci_a")

        text = (
            f"Across {n_sims:,} stochastic iterations, {fighter_a} registers a {p_a}% projected win rate, "
            f"while {fighter_b} sits at {p_b}% with an average finish interval of {gt['avg_finish_interval']}."
        )
        claims = [
            Claim(type="fighter", subject=fighter_a),
            Claim(type="fighter", subject=fighter_b),
            Claim(type="win_prob", subject=fighter_a, value=p_a),
            Claim(type="win_prob", subject=fighter_b, value=p_b),
            Claim(type="provenance", subject="n_simulations", value=n_sims),
        ]
        if ci:
            claims.append(Claim(type="ci", subject=fighter_a, lo=round(ci[0] * 100, 1), hi=round(ci[1] * 100, 1)))
        if gt.get("method_breakdown"):
            method, share = max(gt["method_breakdown"].items(), key=lambda kv: kv[1])
            claims.append(Claim(type="method_share", subject=method, value=round(share * 100, 1)))

        if attack == "win_rate":  # the original lie: wrong number in text AND claims
            return Narration(
                text=f"Tactical brief: {fighter_a} has a dominant {fake}% win rate over {fighter_b}.",
                claims=[Claim(type="win_prob", subject=fighter_a, value=fake)],
            )
        if attack == "swapped":  # numbers are real but attached to the wrong fighter
            claims[2], claims[3] = (
                Claim(type="win_prob", subject=fighter_a, value=p_b),
                Claim(type="win_prob", subject=fighter_b, value=p_a),
            )
        elif attack == "bad_ci":
            claims.append(Claim(type="ci", subject=fighter_b, lo=1.0, hi=99.0))
        elif attack == "fake_method":
            claims.append(Claim(type="method_share", subject="Flying Armbar", value=40.0))
        elif attack == "no_claims":
            claims = []
        elif attack == "text_lie":  # claims are honest but the prose lies
            text = f"{fighter_a} wins {fake}% of simulations."
        return Narration(text=text, claims=claims)

    def _narrator(self, attack: str | None):
        """Builds the narrator callable the harness will use (mock or real LLM)."""

        def narrate(ground_truth: dict[str, Any], fighter_a: str, fighter_b: str, violations: list[str]) -> Narration:
            p_a = round(ground_truth["win_rate_a"] * 100, 1)
            p_b = round(ground_truth["win_rate_b"] * 100, 1)

            if self.use_mock or self.client is None:
                return self._mock_narration(ground_truth, fighter_a, fighter_b, attack)

            prompt = (
                f"Ground truth telemetry: {ground_truth}. Write a 2-sentence tactical brief stating exact win "
                f"rates ({p_a}% vs {p_b}%). Do not alter any numbers. Reply with ONLY JSON: "
                '{"text": "<brief>", "claims": [{"type": "fighter|win_prob|method_share|ci|provenance", '
                '"subject": "<fighter, method, or n_simulations>", "value": <percent or count>, '
                '"lo": <percent>, "hi": <percent>}]} listing every number or name your text states.'
            )
            if violations:
                prompt += f" Your previous draft failed an audit: {violations}. Use ONLY the numbers given."
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=400,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            try:
                cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
                return Narration.model_validate(json.loads(cleaned))
            except (ValueError, TypeError):
                # Unparseable output has no checkable claims, so the audit fails and the harness retries.
                return Narration(text=raw, claims=[])

        return narrate

    def execute_fight_analysis(
        self,
        fighter_a: str,
        fighter_b: str,
        user_query: str | None = None,
        force_hallucination: bool | str = False,
        num_simulations: int = 1000,
    ) -> CopilotAnalysisResult:
        harness = AgentHarness(
            narrator=self._narrator("win_rate" if force_hallucination is True else (force_hallucination or None)),
            auditor=self.auditor,
            max_attempts=self.max_attempts,
        )
        return harness.run(fighter_a, fighter_b, user_query, num_simulations)


# Backward compatibility alias
AgenticCopilot = DecisionCopilot
