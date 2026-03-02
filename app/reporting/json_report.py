import json
from pathlib import Path

from app.models.results import EvaluationReport


def write_json_report(report: EvaluationReport, output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")