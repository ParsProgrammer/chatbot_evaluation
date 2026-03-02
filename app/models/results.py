from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class TurnResult(BaseModel):
    turn_index: int
    expected_intent: str
    predicted_intent: str
    intent_correct: bool
    response_pass: bool
    latency_ms: Optional[float] = None

    # Rich per-turn metrics (optional, so backwards-compatible)
    confidence: Optional[float] = None
    intent_match_method: Optional[str] = None   # exact/alias/prefix/fuzzy
    intent_match_score: Optional[float] = None  # 0..1
    response_semantic_score: Optional[float] = None  # 0..1 when semantic rule used
    response_rule_hits: Optional[List[str]] = None   # e.g. ["kw:pass", "semantic:pass@0.70", "regex:fail"]


class RunResult(BaseModel):
    run_index: int
    test_id: str
    user_id: str
    turns: List[TurnResult] = Field(default_factory=list)


# -------------------------
# Organized summary models
# -------------------------

class CorrectnessMetrics(BaseModel):
    intent_accuracy: float
    response_pass_rate: float


class SemanticQualityMetrics(BaseModel):
    avg_response_semantic_score: Optional[float] = None


class CalibrationMetrics(BaseModel):
    avg_confidence: Optional[float] = None


class PerformanceMetrics(BaseModel):
    avg_latency_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p90_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None


class StabilityMetrics(BaseModel):
    intent_agreement_rate: Optional[float] = None
    response_agreement_rate: Optional[float] = None


class MajorityFailureTurn(BaseModel):
    turn_index: int
    kind: str  # "intent" or "response"
    true_votes: int
    false_votes: int
    majority_passed: bool


class MajorityFailureBreakdown(BaseModel):
    """
    Per test_id, which turns failed by majority voting and why.
    """
    failures_by_test: Dict[str, List[MajorityFailureTurn]] = Field(default_factory=dict)


class ReportSummary(BaseModel):
    total_tests: int
    total_turns: int
    runs: int
    failed_test_ids: List[str]

    correctness: CorrectnessMetrics
    semantic_quality: SemanticQualityMetrics = Field(default_factory=SemanticQualityMetrics)
    calibration: CalibrationMetrics = Field(default_factory=CalibrationMetrics)
    performance: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    stability: StabilityMetrics = Field(default_factory=StabilityMetrics)

    majority_failure_breakdown: MajorityFailureBreakdown = Field(default_factory=MajorityFailureBreakdown)


class EvaluationReport(BaseModel):
    summary: ReportSummary
    # Optional but highly useful: include raw results for per-turn rule-level metrics
    results: List[RunResult] = Field(default_factory=list)