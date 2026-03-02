from collections import defaultdict
from typing import List, Dict, Optional, Tuple

from app.models.results import (
    RunResult,
    EvaluationReport,
    ReportSummary,
    CorrectnessMetrics,
    SemanticQualityMetrics,
    CalibrationMetrics,
    PerformanceMetrics,
    StabilityMetrics,
    MajorityFailureBreakdown,
    MajorityFailureTurn,
)


def _majority(votes: List[bool]) -> bool:
    # ties fail (conservative)
    t = sum(1 for v in votes if v)
    f = len(votes) - t
    return t > f


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if p <= 0:
        return float(vals[0])
    if p >= 100:
        return float(vals[-1])
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[f])
    return float(vals[f] + (vals[c] - vals[f]) * (k - f))


def _agreement_rate(votes_map: Dict[Tuple[str, int], List[bool]]) -> Optional[float]:
    """
    For each (test_id, turn_idx): agreement = max(true,false)/N
    Return average across turns.
    """
    if not votes_map:
        return None
    per_turn = []
    for votes in votes_map.values():
        if not votes:
            continue
        t = sum(1 for v in votes if v)
        f = len(votes) - t
        per_turn.append(max(t, f) / len(votes))
    return (sum(per_turn) / len(per_turn)) if per_turn else None


def aggregate_report(run_results: List[RunResult], runs: int) -> EvaluationReport:
    # Group by test_id then turn_index
    by_test: Dict[str, List[RunResult]] = defaultdict(list)
    for rr in run_results:
        by_test[rr.test_id].append(rr)

    total_tests = len(by_test)
    total_turns_all_runs = 0
    total_intent_correct_all_runs = 0
    total_response_pass_all_runs = 0

    latencies: List[float] = []
    confidences: List[float] = []
    semantic_scores: List[float] = []

    failed_test_ids: List[str] = []
    breakdown = MajorityFailureBreakdown(failures_by_test={})

    # For stability metrics (agreement across runs)
    intent_votes_by_turn: Dict[Tuple[str, int], List[bool]] = defaultdict(list)
    resp_votes_by_turn: Dict[Tuple[str, int], List[bool]] = defaultdict(list)

    for test_id, results_for_test in by_test.items():
        max_turns = max((len(r.turns) for r in results_for_test), default=0)
        test_failed = False
        failures_for_test: List[MajorityFailureTurn] = []

        for turn_idx in range(max_turns):
            intent_votes: List[bool] = []
            resp_votes: List[bool] = []

            for rr in results_for_test:
                if turn_idx < len(rr.turns):
                    tr = rr.turns[turn_idx]
                    intent_votes.append(tr.intent_correct)
                    resp_votes.append(tr.response_pass)

                    total_turns_all_runs += 1
                    total_intent_correct_all_runs += 1 if tr.intent_correct else 0
                    total_response_pass_all_runs += 1 if tr.response_pass else 0

                    if tr.latency_ms is not None:
                        latencies.append(float(tr.latency_ms))
                    if tr.confidence is not None:
                        confidences.append(float(tr.confidence))
                    if tr.response_semantic_score is not None:
                        semantic_scores.append(float(tr.response_semantic_score))

            if intent_votes:
                intent_votes_by_turn[(test_id, turn_idx)].extend(intent_votes)
            if resp_votes:
                resp_votes_by_turn[(test_id, turn_idx)].extend(resp_votes)

            # Majority logic per requirement: test fails if any turn fails in majority of runs
            if intent_votes:
                maj_ok = _majority(intent_votes)
                if not maj_ok:
                    test_failed = True
                    t = sum(1 for v in intent_votes if v)
                    f = len(intent_votes) - t
                    failures_for_test.append(
                        MajorityFailureTurn(
                            turn_index=turn_idx,
                            kind="intent",
                            true_votes=t,
                            false_votes=f,
                            majority_passed=False,
                        )
                    )

            if resp_votes:
                maj_ok = _majority(resp_votes)
                if not maj_ok:
                    test_failed = True
                    t = sum(1 for v in resp_votes if v)
                    f = len(resp_votes) - t
                    failures_for_test.append(
                        MajorityFailureTurn(
                            turn_index=turn_idx,
                            kind="response",
                            true_votes=t,
                            false_votes=f,
                            majority_passed=False,
                        )
                    )

        if test_failed:
            failed_test_ids.append(test_id)
            breakdown.failures_by_test[test_id] = failures_for_test

    intent_accuracy = (total_intent_correct_all_runs / total_turns_all_runs) if total_turns_all_runs else 0.0
    response_pass_rate = (total_response_pass_all_runs / total_turns_all_runs) if total_turns_all_runs else 0.0

    avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else None
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else None
    avg_semantic = (sum(semantic_scores) / len(semantic_scores)) if semantic_scores else None

    perf = PerformanceMetrics(
        avg_latency_ms=(round(avg_latency_ms, 2) if avg_latency_ms is not None else None),
        latency_p50_ms=(round(_percentile(latencies, 50), 2) if latencies else None),
        latency_p90_ms=(round(_percentile(latencies, 90), 2) if latencies else None),
        latency_p99_ms=(round(_percentile(latencies, 99), 2) if latencies else None),
    )

    summary = ReportSummary(
        total_tests=total_tests,
        total_turns=total_turns_all_runs,
        runs=max(1, runs),
        failed_test_ids=sorted(failed_test_ids),

        correctness=CorrectnessMetrics(
            intent_accuracy=round(intent_accuracy, 4),
            response_pass_rate=round(response_pass_rate, 4),
        ),
        semantic_quality=SemanticQualityMetrics(
            avg_response_semantic_score=(round(avg_semantic, 4) if avg_semantic is not None else None),
        ),
        calibration=CalibrationMetrics(
            avg_confidence=(round(avg_confidence, 4) if avg_confidence is not None else None),
        ),
        performance=perf,
        stability=StabilityMetrics(
            intent_agreement_rate=(round(_agreement_rate(intent_votes_by_turn), 4) if intent_votes_by_turn else None),
            response_agreement_rate=(round(_agreement_rate(resp_votes_by_turn), 4) if resp_votes_by_turn else None),
        ),
        majority_failure_breakdown=breakdown,
    )

    # IMPORTANT: include raw results so per-turn rule-level metrics show up in report.json
    return EvaluationReport(summary=summary, results=run_results)