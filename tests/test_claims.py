"""Slice 4 (Red): structured claims, auditor, harness wiring, one tolerance, Wilson CIs."""
import pytest
from pydantic import ValidationError

GT = {
    "fighter_a": "Ali Test", "fighter_b": "Bo Example", "total_simulations": 200,
    "win_rate_a": 0.62, "win_rate_b": 0.38, "avg_finish_interval": 7.5,
    "win_ci_a": [0.55, 0.69], "win_ci_b": [0.31, 0.45],
    "method_breakdown": {"KO/TKO": 0.40, "Submission": 0.25, "Decision": 0.35},
}


def good_claims():
    from src.agent.schemas import Claim

    return [
        Claim(type="fighter", subject="Ali Test"),
        Claim(type="win_prob", subject="Ali Test", value=62.0),
        Claim(type="win_prob", subject="Bo Example", value=38.0),
        Claim(type="method_share", subject="KO/TKO", value=40.0),
        Claim(type="ci", subject="Ali Test", lo=55.0, hi=69.0),
        Claim(type="provenance", subject="n_simulations", value=200),
    ]


def audit(claims, tol=None):
    from src.agent.auditor import AnalyticalAuditor

    return AnalyticalAuditor().verify_structured(claims, GT, tolerance_pct=tol)


def test_claim_rejects_unknown_type():
    from src.agent.schemas import Claim

    with pytest.raises(ValidationError):
        Claim(type="vibes", subject="x", value=1.0)


def test_all_correct_claims_pass():
    ok, violations = audit(good_claims())
    assert ok and violations == []


def test_win_prob_within_tolerance_passes_and_outside_fails():
    from src.agent.schemas import Claim

    assert audit([Claim(type="win_prob", subject="Ali Test", value=63.4)])[0]
    assert not audit([Claim(type="win_prob", subject="Ali Test", value=63.6)])[0]


def test_swapped_fighters_are_caught():
    from src.agent.schemas import Claim

    ok, violations = audit([Claim(type="win_prob", subject="Ali Test", value=38.0)])
    assert not ok and "Ali Test" in violations[0]  # regex-only auditing would have passed this


def test_wrong_ci_bound_fails():
    from src.agent.schemas import Claim

    assert not audit([Claim(type="ci", subject="Ali Test", lo=50.0, hi=69.0)])[0]


def test_unknown_fighter_unknown_method_and_wrong_provenance_fail():
    from src.agent.schemas import Claim

    assert not audit([Claim(type="fighter", subject="Charles Hallucinated")])[0]
    assert not audit([Claim(type="method_share", subject="Flying Armbar", value=40.0)])[0]
    assert not audit([Claim(type="provenance", subject="n_simulations", value=1000)])[0]


def test_empty_claims_are_not_verifiable():
    ok, violations = audit([])
    assert not ok and "no verifiable claims" in violations[0].lower()


def fake_sim(a, b, num_simulations=1000):
    return dict(GT)


def run_with(narration):
    from src.agent.harness import AgentHarness

    return AgentHarness(lambda gt, a, b, v: narration, simulate=fake_sim).run("Ali Test", "Bo Example")


def test_harness_accepts_structured_narration():
    from src.agent.schemas import Narration

    result = run_with(Narration(text="Ali Test wins 62.0% of simulations.", claims=good_claims()))
    assert result.status == "success"


def test_harness_refuses_when_claim_is_wrong_even_if_text_looks_fine():
    from src.agent.schemas import Claim, Narration

    bad = [Claim(type="win_prob", subject="Ali Test", value=38.0)]
    result = run_with(Narration(text="Ali Test wins 62.0% of simulations.", claims=bad))
    assert result.status == "refused" and result.violations


def test_harness_refuses_when_text_hallucinates_even_if_claims_are_right():
    from src.agent.schemas import Narration

    result = run_with(Narration(text="Ali Test has an overwhelming 98.5% win rate.", claims=good_claims()))
    assert result.status == "refused"


def test_single_audit_tolerance_constant():
    from src.agent.auditor import AnalyticalAuditor
    from src.agent.copilot import DecisionCopilot
    from src.engine import app_config

    assert app_config.AUDIT_TOLERANCE_PCT == 1.5
    assert AnalyticalAuditor().tolerance_pct == app_config.AUDIT_TOLERANCE_PCT
    assert DecisionCopilot(use_mock=True).tolerance_pct == app_config.AUDIT_TOLERANCE_PCT


def test_wilson_interval_known_value_and_edges():
    from src.engine.intervals import wilson_interval

    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo0, hi0 = wilson_interval(0, 100)
    assert lo0 == pytest.approx(0.0, abs=1e-9) and hi0 > 0.0
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_simulation_response_carries_confidence_intervals():
    from src.agent.schemas import SimulationResponse

    resp = SimulationResponse(
        fighter_a="A", fighter_b="B", total_simulations=200, win_rate_a=0.6, win_rate_b=0.4,
        draw_rate=0.0, avg_finish_interval=7.0, method_breakdown={"Decision": 1.0},
        win_ci_a=(0.53, 0.67), win_ci_b=(0.33, 0.47),
    )
    assert resp.model_dump()["win_ci_a"] == (0.53, 0.67)
