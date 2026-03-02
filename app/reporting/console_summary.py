# app/reporting/console_summary.py
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from app.models.results import EvaluationReport, RunResult, TurnResult


def _index_results(report: EvaluationReport) -> Dict[str, List[RunResult]]:
    by_test: Dict[str, List[RunResult]] = defaultdict(list)
    for rr in report.results:
        by_test[rr.test_id].append(rr)
    return by_test


def _stats(vals: List[float]) -> Optional[Tuple[float, float, float]]:
    if not vals:
        return None
    vals = sorted(vals)
    avg = sum(vals) / len(vals)
    return avg, vals[0], vals[-1]


def _fmt_stats(vals: List[float], nd: int = 3) -> str:
    st = _stats(vals)
    if not st:
        return "n/a"
    avg, mn, mx = st
    return f"{avg:.{nd}f} (min {mn:.{nd}f}, max {mx:.{nd}f})"


def _fmt_counter(c: Counter, limit: int = 5) -> str:
    items = c.most_common(limit)
    if not items:
        return "n/a"
    s = ", ".join([f"{k}×{v}" for k, v in items])
    if len(c) > limit:
        s += ", …"
    return s


def _short(s: str, n: int = 70) -> str:
    t = (s or "").replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def print_console_summary(report: EvaluationReport) -> None:
    s = report.summary

    print("\n=== Chatbot Evaluation Summary ===")
    print(f"Total Tests: {s.total_tests}")
    print(f"Total Turns (across runs): {s.total_turns}")
    print(f"Runs: {s.runs}")

    print("\n--- Correctness ---")
    print(f"Intent Accuracy: {s.correctness.intent_accuracy * 100:.2f}%")
    print(f"Response Pass Rate: {s.correctness.response_pass_rate * 100:.2f}%")

    print("\n--- Semantic Quality ---")
    if s.semantic_quality.avg_response_semantic_score is not None:
        print(f"Avg Response Semantic Score (where used): {s.semantic_quality.avg_response_semantic_score:.4f}")
    else:
        print("Avg Response Semantic Score: n/a (no semantic checks used or no scores captured)")

    print("\n--- Calibration ---")
    if s.calibration.avg_confidence is not None:
        print(f"Average Confidence: {s.calibration.avg_confidence:.4f}")
    else:
        print("Average Confidence: n/a")

    print("\n--- Performance ---")
    if s.performance.avg_latency_ms is not None:
        print(f"Average Latency: {s.performance.avg_latency_ms:.2f} ms")
    if (
        s.performance.latency_p50_ms is not None
        and s.performance.latency_p90_ms is not None
        and s.performance.latency_p99_ms is not None
    ):
        print(
            f"Latency p50/p90/p99: "
            f"{s.performance.latency_p50_ms:.2f} / {s.performance.latency_p90_ms:.2f} / {s.performance.latency_p99_ms:.2f} ms"
        )

    print("\n--- Stability ---")
    if s.stability.intent_agreement_rate is not None:
        print(f"Intent Agreement Rate: {s.stability.intent_agreement_rate * 100:.2f}%")
    if s.stability.response_agreement_rate is not None:
        print(f"Response Agreement Rate: {s.stability.response_agreement_rate * 100:.2f}%")

    if not s.failed_test_ids:
        print("\nNo failed tests 🎉")
        return

    print("\n--- Majority-of-Runs Failures ---")
    print("Failed Tests:")
    for tid in s.failed_test_ids:
        print(f" - {tid}")

    by_test_runs = _index_results(report)

    # -----------------------------
    # MORE INFORMATIVE BREAKDOWN
    # -----------------------------
    print("\nFailure Breakdown (by test, turn):")

    for tid, failures in s.majority_failure_breakdown.failures_by_test.items():
        # group majority failures by turn, keep intent/response together
        by_turn: Dict[int, Dict[str, object]] = defaultdict(dict)
        for f in failures:
            by_turn[int(f.turn_index)][str(f.kind)] = f

        print(f"\n - {tid}")

        test_runs = by_test_runs.get(tid, [])

        for turn_idx in sorted(by_turn.keys()):
            intent_fail_item = by_turn[turn_idx].get("intent")
            resp_fail_item = by_turn[turn_idx].get("response")

            intent_failed = intent_fail_item is not None
            resp_failed = resp_fail_item is not None

            intent_votes = (
                f"{intent_fail_item.true_votes}/{intent_fail_item.false_votes}" if intent_fail_item else "-"
            )
            resp_votes = (
                f"{resp_fail_item.true_votes}/{resp_fail_item.false_votes}" if resp_fail_item else "-"
            )

            # gather TurnResults across runs for this turn
            trs: List[TurnResult] = []
            for rr in test_runs:
                if turn_idx < len(rr.turns):
                    trs.append(rr.turns[turn_idx])

            # Aggregates from TurnResults (best-effort)
            expected_intent = trs[0].expected_intent if trs else "unknown"

            pred_intents = Counter([(tr.predicted_intent or "∅") for tr in trs])
            match_methods = Counter([(tr.intent_match_method or "∅") for tr in trs])

            intent_scores = [float(tr.intent_match_score) for tr in trs if tr.intent_match_score is not None]
            sem_scores = [float(tr.response_semantic_score) for tr in trs if tr.response_semantic_score is not None]
            confidences = [float(tr.confidence) for tr in trs if tr.confidence is not None]
            latencies = [float(tr.latency_ms) for tr in trs if tr.latency_ms is not None]

            # rule hits (response)
            rule_hits = Counter()
            for tr in trs:
                if tr.response_rule_hits:
                    rule_hits.update(tr.response_rule_hits)

            # one clean headline line per turn
            i_mark = "✗" if intent_failed else "✓"
            r_mark = "✗" if resp_failed else "✓"

            # include avg scores inline if present
            intent_avg = (_stats(intent_scores)[0] if _stats(intent_scores) else None)
            sem_avg = (_stats(sem_scores)[0] if _stats(sem_scores) else None)

            intent_avg_str = f"{intent_avg:.3f}" if intent_avg is not None else "n/a"
            sem_avg_str = f"{sem_avg:.3f}" if sem_avg is not None else "n/a"

            print(
                f"    turn {turn_idx:<2} | "
                f"intent {i_mark} (votes {intent_votes}, avg score {intent_avg_str}) | "
                f"response {r_mark} (votes {resp_votes}, avg sem {sem_avg_str})"
            )

            # details (only the useful stuff, but still compact)
            print(f"      expected_intent: {expected_intent}")
            if trs:
                print(f"      predicted_intent: {_fmt_counter(pred_intents, limit=5)}")
                print(f"      intent_method:    {_fmt_counter(match_methods, limit=5)}")

            if intent_failed:
                print(f"      intent_score:     {_fmt_stats(intent_scores, nd=3)}")

            if resp_failed:
                print(f"      semantic_score:   {_fmt_stats(sem_scores, nd=3)}")
                top_hits = rule_hits.most_common(8)
                if top_hits:
                    hits_str = ", ".join([f"{_short(k)}×{v}" for k, v in top_hits])
                else:
                    hits_str = "n/a"
                print(f"      top_rule_hits:    {hits_str}")

            # extra turn-level metrics that are great for debugging but not too noisy
            if confidences:
                print(f"      confidence(avg):  {sum(confidences)/len(confidences):.3f}")
            if latencies:
                print(f"      latency(avg ms):  {sum(latencies)/len(latencies):.1f}")