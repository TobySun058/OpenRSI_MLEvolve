"""Compare Qwen and Frontis with the structured local smoke test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "run_frontis_smoke.py"


def _run_smoke(model: str, base_url: str, api_key: str, result_file: Path) -> dict:
    command = [
        sys.executable,
        str(SMOKE),
        "--model",
        model,
        "--base-url",
        base_url,
        "--api-key",
        api_key,
        "--result-file",
        str(result_file),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    output = (completed.stdout + completed.stderr).strip()
    print(f"\n===== {model} =====")
    print(output)
    if result_file.exists():
        data = json.loads(result_file.read_text(encoding="utf-8"))
    else:
        data = {
            "model": model,
            "full_loop_completed": False,
            "draft": {"structured_output_success": False, "python_valid": False, "execution_success": False, "submission_valid": False, "metric": None},
            "feedback": {"structured_output_success": False},
            "improvement": {"structured_output_success": False, "python_valid": False, "execution_success": False, "submission_valid": False, "metric": None},
            "metric_delta": None,
        }
    data["returncode"] = completed.returncode
    data["output"] = output
    return data


def _flag(v: bool) -> str:
    return "PASS" if v else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", default="Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--base-url-a", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model-b", default="Frontis-MA1-30B")
    parser.add_argument("--base-url-b", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--api-key-a", default="EMPTY")
    parser.add_argument("--api-key-b", default="EMPTY")
    parser.add_argument("--report", type=Path, default=ROOT / "runs" / "model_comparison_structured.json")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mlevolve_compare_") as tmpdir:
        tmp = Path(tmpdir)
        result_a = _run_smoke(args.model_a, args.base_url_a, args.api_key_a, tmp / "qwen.json")
        result_b = _run_smoke(args.model_b, args.base_url_b, args.api_key_b, tmp / "frontis.json")

    report = {
        "model_a": result_a,
        "model_b": result_b,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"\nReport: {args.report}")
    print("\nMetric Summary")
    print(f"{'Stage':<24}{args.model_a:<16}{args.model_b:<16}")
    rows = [
        ("Draft JSON", result_a["draft"]["structured_output_success"], result_b["draft"]["structured_output_success"]),
        ("Draft Python", result_a["draft"]["python_valid"], result_b["draft"]["python_valid"]),
        ("Draft Exec", result_a["draft"]["execution_success"], result_b["draft"]["execution_success"]),
        ("Draft Valid", result_a["draft"]["submission_valid"], result_b["draft"]["submission_valid"]),
        ("Feedback JSON", result_a["feedback"]["structured_output_success"], result_b["feedback"]["structured_output_success"]),
        ("Improve JSON", result_a["improvement"]["structured_output_success"], result_b["improvement"]["structured_output_success"]),
        ("Improve Python", result_a["improvement"]["python_valid"], result_b["improvement"]["python_valid"]),
        ("Improve Exec", result_a["improvement"]["execution_success"], result_b["improvement"]["execution_success"]),
        ("Improve Valid", result_a["improvement"]["submission_valid"], result_b["improvement"]["submission_valid"]),
        ("Full Loop", result_a["full_loop_completed"], result_b["full_loop_completed"]),
    ]
    for stage, a_val, b_val in rows:
        print(f"{stage:<24}{_flag(bool(a_val)):<16}{_flag(bool(b_val)):<16}")
    print(f"{'Draft Metric':<24}{str(result_a['draft'].get('metric')):<16}{str(result_b['draft'].get('metric')):<16}")
    print(f"{'Improve Metric':<24}{str(result_a['improvement'].get('metric')):<16}{str(result_b['improvement'].get('metric')):<16}")
    print(f"{'Metric Delta':<24}{str(result_a.get('metric_delta')):<16}{str(result_b.get('metric_delta')):<16}")

    return 0 if result_a.get("full_loop_completed") and result_b.get("full_loop_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
