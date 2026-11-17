"""Pydantic contracts for the agent tools and the Copilot's structured outputs."""

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class MetricQueryRequest(BaseModel):
    fighter_name: str = Field(..., description="Full legal name of the fighter.")
    metric_name: str = Field(..., description="Semantic metric key defined in metrics.yml (e.g. ATT, DEF, ufc_bouts).")


class MetricQueryResponse(BaseModel):
    fighter_name: str
    metric_name: str
    value: float
    status: str = "governed_success"


class SimulationRequest(BaseModel):
    fighter_a: str = Field(..., description="First fighter name.")
    fighter_b: str = Field(..., description="Second fighter name.")
    num_simulations: int = Field(default=1000, ge=100, le=10000, description="Monte Carlo sample count.")


class SimulationResponse(BaseModel):
    fighter_a: str
    fighter_b: str
    total_simulations: int
    win_rate_a: float
    win_rate_b: float
    draw_rate: float
    avg_finish_interval: float
    method_breakdown: Dict[str, float]
    win_ci_a: Optional[Tuple[float, float]] = None  # 95% Wilson interval, fractions
    win_ci_b: Optional[Tuple[float, float]] = None


class Claim(BaseModel):
    """One checkable statement the narrator makes (an 'itemized receipt' line)."""

    type: Literal["fighter", "win_prob", "method_share", "ci", "provenance"]
    subject: str  # fighter name, method name, or provenance key (e.g. "n_simulations")
    value: Optional[float] = None  # percent for win_prob/method_share; raw count for provenance
    lo: Optional[float] = None  # percent bounds for ci
    hi: Optional[float] = None


class Narration(BaseModel):
    """What a narrator returns: prose for humans plus claims for the auditor."""

    text: str
    claims: List[Claim] = Field(default_factory=list)


class CopilotAnalysisResult(BaseModel):
    status: str  # "success", "refused", or "error"
    narrative: str
    violations: List[str] = Field(default_factory=list)
    ground_truth: Optional[Dict] = None
    attempts: int = 0  # narration attempts made by the harness
    trace: List[str] = Field(default_factory=list)  # plain-English audit trail of each harness step
