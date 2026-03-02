import argparse
import asyncio
import json
from pathlib import Path

from app.client.fallback_client import FallbackChatClient
from app.evaluator.runner import EvaluationRunner
from app.evaluator.metrics import aggregate_report
from app.models.dataset import Dataset
from app.reporting.console_summary import print_console_summary
from app.reporting.json_report import write_json_report


async def async_main(args: argparse.Namespace) -> int:
    dataset_obj = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    dataset = Dataset.from_json_obj(dataset_obj)
    tests = [t.normalized() for t in dataset.tests]

    client = FallbackChatClient(
        base_url=args.base_url,
        timeout_s=args.timeout,
        verbose=True,
    )

    try:
        runner = EvaluationRunner(client=client, concurrency=args.concurrency)
        run_results = await runner.run_dataset(tests=tests, runs=args.runs)

        report = aggregate_report(run_results=run_results, runs=args.runs)

        print_console_summary(report)
        write_json_report(report, args.output)
        return 0
    finally:
        await client.aclose()


def main() -> int:
    p = argparse.ArgumentParser(description="Automated Chatbot Evaluation Pipeline")
    p.add_argument("--dataset", required=True, help="Path to test dataset JSON")
    p.add_argument("--base-url", required=True, help="Chatbot base URL, e.g. http://localhost:8080")
    p.add_argument("--runs", type=int, default=1, help="Number of runs (LLM variability)")
    p.add_argument("--output", required=True, help="Output report.json path")
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds")
    p.add_argument("--concurrency", type=int, default=10, help="Max concurrent test executions")
    args = p.parse_args()

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())