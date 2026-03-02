import asyncio
import uuid
from typing import List, Optional

from app.client.base import ChatClient
from app.evaluator.validator import intent_evaluate, response_evaluate
from app.models.dataset import TestCase
from app.models.results import RunResult, TurnResult


class EvaluationRunner:
    """
    - Runs each conversation sequentially (preserves state via user_id)
    - Runs different test cases concurrently (each test gets its own user_id)
    """

    def __init__(self, client: ChatClient, concurrency: int = 10, fail_fast: bool = True):
        self.client = client
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self.fail_fast = fail_fast

    async def _ensure_client_ready(self) -> None:
        ensure = getattr(self.client, "ensure_ready", None)
        if callable(ensure):
            await ensure()

    def _make_user_id(self, test_id: str, run_index: int) -> str:
        return f"{test_id}-run{run_index}-{uuid.uuid4().hex[:8]}"

    async def run_test_case(self, test: TestCase, run_index: int) -> RunResult:
        test = test.normalized()
        user_id = self._make_user_id(test.test_id, run_index)

        rr = RunResult(run_index=run_index, test_id=test.test_id, user_id=user_id, turns=[])

        for i, user_msg in enumerate(test.conversation):
            expected_intent = test.expected_intents[i] if i < len(test.expected_intents) else "unknown"
            expected_keywords = test.expected_response_keywords[i] if i < len(test.expected_response_keywords) else []

            resp_json, latency_ms = await self.client.chat(user_id=user_id, message=user_msg)

            predicted_intent = str(resp_json.get("intent", ""))
            response_text = str(resp_json.get("response", ""))

            confidence = resp_json.get("confidence", None)
            try:
                confidence = float(confidence) if confidence is not None else None
            except Exception:
                confidence = None

            i_eval = intent_evaluate(predicted_intent, expected_intent)
            r_eval = response_evaluate(response_text, expected_keywords)

            rr.turns.append(
                TurnResult(
                    turn_index=i,
                    expected_intent=expected_intent,
                    predicted_intent=predicted_intent,
                    intent_correct=bool(i_eval["passed"]),
                    response_pass=bool(r_eval["passed"]),
                    latency_ms=latency_ms,

                    confidence=confidence,
                    intent_match_method=str(i_eval.get("method")) if i_eval.get("method") else None,
                    intent_match_score=float(i_eval.get("score")) if i_eval.get("score") is not None else None,
                    response_semantic_score=float(r_eval.get("semantic_score")) if r_eval.get("semantic_score") is not None else None,
                    response_rule_hits=list(r_eval.get("rule_hits")) if r_eval.get("rule_hits") is not None else None,
                )
            )

        return rr

    async def run_dataset(self, tests: List[TestCase], runs: int) -> List[RunResult]:
        await self._ensure_client_ready()

        tests = [t.normalized() for t in tests]
        runs = max(1, runs)

        async def _guarded(test: TestCase, run_idx: int) -> Optional[RunResult]:
            async with self._sem:
                try:
                    return await self.run_test_case(test, run_idx)
                except Exception:
                    if self.fail_fast:
                        raise
                    return None

        tasks = [asyncio.create_task(_guarded(t, r)) for r in range(runs) for t in tests]

        results: List[RunResult] = []
        for coro in asyncio.as_completed(tasks):
            rr = await coro
            if rr is not None:
                results.append(rr)

        return results