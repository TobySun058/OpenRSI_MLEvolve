"""Standalone structured integration smoke test for Qwen/Frontis.

This does not run native MLEvolve search; it only probes endpoint behavior
and a tiny local plan/code/execute/review loop to isolate API compatibility.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import prep_agent_workspace
from engine.executor import Interpreter
from engine.search_node import SearchNode
from utils.logging_config import setup_logging
from utils.metric import WorstMetricValue


def _make_cfg(base_dir: Path, data_dir: Path, desc_file: Path, model: str, base_url: str, api_key: str):
    agent = SimpleNamespace(
        steps=2,
        time_limit=300,
        initial_drafts=1,
        seed=42,
        data_preview=True,
        code=SimpleNamespace(model=model, temp=0.2, base_url=base_url, api_key=api_key),
        feedback=SimpleNamespace(model=model, temp=0.2, base_url=base_url, api_key=api_key),
        search=SimpleNamespace(parallel_search_num=1, num_gpus=0),
        check_data_leakage=False,
        fusion_vs_evolution_prob=0.3,
        branch_fusion_trigger_prob=1.0,
        max_fusion_drafts=0,
        use_global_memory=False,
        memory_similarity_threshold=0.7,
        memory_embedding_device="cpu",
        memory_embedding_model_path="",
        use_diff_mode=False,
        use_stepwise_generation=False,
        use_evolution=False,
        use_fusion=False,
        use_aggregation=False,
    )
    return SimpleNamespace(
        data_dir=data_dir,
        dataset_dir=base_dir,
        desc_file=desc_file,
        goal=None,
        eval=None,
        log_dir=base_dir / "runs" / "frontis_smoke",
        log_level="INFO",
        workspace_dir=base_dir / "runs" / "frontis_smoke",
        preprocess_data=False,
        copy_data=True,
        exp_name="frontis_smoke",
        exp_id="frontis_smoke",
        torch_hub_dir="",
        pretrain_model_dir="",
        exec=SimpleNamespace(timeout=300, agent_file_name="runfile.py"),
        agent=agent,
        start_cpu_id=0,
        cpu_number=1,
        coldstart=SimpleNamespace(use_coldstart=False, task_json_path="", model_json_path="", description=""),
        use_grading_server=False,
        init_solution=SimpleNamespace(use=False),
    )


def _client(args: argparse.Namespace) -> OpenAI:
    return OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=1200)


def _json_call(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
        extra_body={"think": False},
    )
    raw = resp.choices[0].message.content or ""
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"failed to parse JSON output: {exc}\nRAW:\n{raw}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object, got {type(payload).__name__}\nRAW:\n{raw}")
    return payload, raw


def _validate_solution(payload: dict[str, Any]) -> tuple[bool, str]:
    plan = payload.get("plan")
    code = payload.get("code")
    if not isinstance(plan, str) or not plan.strip():
        return False, "missing or empty plan"
    if not isinstance(code, str) or not code.strip():
        return False, "missing or empty code"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"python syntax error: {exc}"
    return True, ""


def _execute_code(cfg, code: str, node_id: str) -> tuple[Any, SearchNode, str]:
    interpreter = Interpreter(
        cfg.workspace_dir,
        **{"timeout": cfg.exec.timeout, "agent_file_name": cfg.exec.agent_file_name},
        cfg=cfg,
    )
    node = SearchNode(
        code=code,
        plan="smoke",
        parent=None,
        stage="draft",
        metric=WorstMetricValue(),
        is_buggy=False,
        is_valid=True,
        step=0,
    )
    result = interpreter.run(code, node_id, True)
    node.absorb_exec_result(result)
    return result, node, node_id


def _read_submission(path: Path) -> tuple[bool, list[float] | None, str | None]:
    if not path.exists():
        return False, None, "submission file missing"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False, None, "submission is empty"
    if set(rows[0]) != {"id", "prediction"}:
        return False, None, "submission must contain id,prediction columns"
    preds: list[float] = []
    try:
        for row in rows:
            pred = float(row["prediction"])
            if not math.isfinite(pred):
                return False, None, "prediction is not finite"
            preds.append(pred)
    except Exception as exc:
        return False, None, f"prediction parse error: {exc}"
    return True, preds, None


def _evaluate_submission(
    path: Path,
    expected_ids: list[int],
    targets: list[float],
    execution_success: bool,
) -> dict[str, Any]:
    ok, preds, error = _read_submission(path)
    if not ok:
        return {
            "execution_success": execution_success,
            "submission_valid": False,
            "metric": None,
            "error": error,
            "exc_type": None,
            "exc_info": None,
            "terminal_output": error,
        }
    assert preds is not None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected_ids):
        return {
            "execution_success": execution_success,
            "submission_valid": False,
            "metric": None,
            "error": "row count mismatch",
            "exc_type": None,
            "exc_info": None,
            "terminal_output": "row count mismatch",
        }
    got_ids = [int(row["id"]) for row in rows]
    if got_ids != expected_ids:
        return {
            "execution_success": execution_success,
            "submission_valid": False,
            "metric": None,
            "error": "id mismatch",
            "exc_type": None,
            "exc_info": None,
            "terminal_output": "id mismatch",
        }
    mse = statistics.fmean((p - t) ** 2 for p, t in zip(preds, targets))
    return {
        "execution_success": execution_success,
        "submission_valid": True,
        "metric": mse,
        "error": None,
        "exc_type": None,
        "exc_info": None,
        "terminal_output": None,
    }


def _build_task(base_dir: Path) -> tuple[Path, list[int], list[float]]:
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    desc_file = data_dir / "description.md"
    desc_file.write_text(
        "# Task goal\n"
        "Build a tiny one-feature linear-regression solution.\n\n"
        "# Evaluation\n"
        "Train.csv has columns `id`, `x`, `y`; test.csv has columns `id`, `x`.\n"
        "Write `submission/submission.csv` with the original test `id` values and `prediction`.\n"
        "Use only the Python standard library.\n"
        "The hidden test labels follow the same linear rule as training.\n",
        encoding="utf-8",
    )

    train_rows = [(1, 0, 1.0), (2, 1, 4.0), (3, 2, 7.0), (4, 3, 10.0), (5, 4, 13.0)]
    test_rows = [(6, 5), (7, 6), (8, 7)]
    hidden_targets = [16.0, 19.0, 22.0]

    with (data_dir / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y"])
        writer.writerows(train_rows)

    with (data_dir / "test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x"])
        writer.writerows(test_rows)

    with (data_dir / "sample_submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prediction"])
        writer.writerows([[row[0], 0.0] for row in test_rows])

    return desc_file, [row[0] for row in test_rows], hidden_targets


def _draft_prompt(task_desc: str) -> tuple[str, str]:
    system = (
        "Return exactly one JSON object with keys `plan` and `code`.\n"
        "`plan`: one short sentence.\n"
        "`code`: 10-20 lines of executable Python using only `csv` and built-ins.\n"
        "No markdown, no extra keys, no explanation.\n"
    )
    user = (
        f"Task description:\n{task_desc}\n\n"
        "Write code that reads `input/train.csv` and `input/test.csv`, fits a one-feature linear model from `x` to `y`, "
        "preserves the test `id` values, and writes `submission/submission.csv`."
    )
    return system, user


def _review_prompt(task_desc: str, draft: dict[str, Any], exec_result: dict[str, Any], stdout: str, stderr: str) -> tuple[str, str]:
    system = (
        "Return exactly one JSON object with keys `is_bug`, `analysis`, and `recommended_change`.\n"
        "Do not invent metrics. Use only the execution information provided.\n"
    )
    user = json.dumps(
        {
            "task_description": task_desc,
            "draft_plan": draft["plan"],
            "draft_code": draft["code"],
            "execution_result": exec_result,
            "stdout": stdout,
            "stderr": stderr,
            "execution_details": {
                "exc_type": exec_result.get("exc_type"),
                "exc_info": exec_result.get("exc_info"),
                "terminal_output": exec_result.get("terminal_output"),
                "submission_valid": exec_result.get("submission_valid"),
                "metric": exec_result.get("metric"),
            },
        },
        ensure_ascii=True,
        indent=2,
    )
    return system, user


def _improve_prompt(task_desc: str, draft: dict[str, Any], review: dict[str, Any], exec_result: dict[str, Any]) -> tuple[str, str]:
    system = (
        "Return exactly one JSON object with keys `plan` and `code`.\n"
        "`plan`: one short sentence.\n"
        "`code`: 10-20 lines of executable Python using only `csv` and built-ins.\n"
        "Use the actual execution result and review to fix the code.\n"
    )
    user = json.dumps(
        {
            "task_description": task_desc,
            "original_plan": draft["plan"],
            "original_code": draft["code"],
            "execution_result": exec_result,
            "review": review,
        },
        ensure_ascii=True,
        indent=2,
    )
    return system, user


def _run_one(model: str, base_url: str, api_key: str, result_file: Path) -> dict[str, Any]:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    tmp = Path(tempfile.mkdtemp(prefix="mlevolve_structured_smoke_"))
    try:
        base_dir = tmp
        desc_file, expected_ids, hidden_targets = _build_task(base_dir)
        cfg = _make_cfg(base_dir, desc_file.parent, desc_file, model, base_url, api_key)
        prep_agent_workspace(cfg)
        logger = setup_logging(cfg)
        task_desc = desc_file.read_text(encoding="utf-8")
        client = _client(argparse.Namespace(model=model, base_url=base_url, api_key=api_key))

        print(f"Endpoint: {base_url}")
        print(f"Model: {model}")

        draft_prompt = _draft_prompt(task_desc)
        draft_payload, draft_raw = _json_call(client, model=model, system=draft_prompt[0], user=draft_prompt[1], max_tokens=2048, temperature=0.0)
        draft_ok, draft_error = _validate_solution(draft_payload)
        if not draft_ok:
            raise ValueError(f"draft validation failed: {draft_error}\nRAW:\n{draft_raw}")
        print("[PASS] draft structured JSON")

        draft_code = draft_payload["code"]
        draft_exec_result, draft_node, draft_submission_tag = _execute_code(cfg, draft_code, "draft")
        draft_stdout = "".join(draft_node._term_out)
        draft_stderr = ""
        draft_submission = cfg.workspace_dir / "submission" / f"submission_{draft_submission_tag}.csv"
        draft_exec_success = draft_exec_result.exc_type is None
        draft_eval = _evaluate_submission(draft_submission, expected_ids, hidden_targets, draft_exec_success)
        print(f"[PASS] draft execution: return={draft_exec_success}")
        print(f"[PASS] draft evaluation: submission_valid={draft_eval['submission_valid']}, metric={draft_eval['metric']}")

        review_sys, review_usr = _review_prompt(task_desc, draft_payload, draft_eval, draft_stdout, draft_stderr)
        review_payload, review_raw = _json_call(client, model=model, system=review_sys, user=review_usr, max_tokens=1024, temperature=0.0)
        if not {"is_bug", "analysis", "recommended_change"}.issubset(review_payload):
            raise ValueError(f"review validation failed\nRAW:\n{review_raw}")
        print("[PASS] feedback structured JSON")

        improve_sys, improve_usr = _improve_prompt(task_desc, draft_payload, review_payload, draft_eval)
        improved_payload, improved_raw = _json_call(client, model=model, system=improve_sys, user=improve_usr, max_tokens=2048, temperature=0.0)
        improved_ok, improved_error = _validate_solution(improved_payload)
        if not improved_ok:
            raise ValueError(f"improvement validation failed: {improved_error}\nRAW:\n{improved_raw}")
        print("[PASS] improvement structured JSON")

        improved_code = improved_payload["code"]
        improved_exec_result, improved_node, improved_submission_tag = _execute_code(cfg, improved_code, "improved")
        improved_stdout = "".join(improved_node._term_out)
        improved_stderr = ""
        improved_submission = cfg.workspace_dir / "submission" / f"submission_{improved_submission_tag}.csv"
        improved_exec_success = improved_exec_result.exc_type is None
        improved_eval = _evaluate_submission(improved_submission, expected_ids, hidden_targets, improved_exec_success)
        print(f"[PASS] improved execution: return={improved_exec_success}")
        print(f"[PASS] improved evaluation: submission_valid={improved_eval['submission_valid']}, metric={improved_eval['metric']}")

        draft_metric = draft_eval["metric"]
        improved_metric = improved_eval["metric"]
        metric_delta = None
        if isinstance(draft_metric, (int, float)) and isinstance(improved_metric, (int, float)):
            metric_delta = draft_metric - improved_metric

        result = {
            "model": model,
            "task": "synthetic linear regression",
            "draft": {
                "structured_output_success": True,
                "python_valid": draft_ok,
                "execution_success": draft_exec_success,
                "submission_valid": draft_eval["submission_valid"],
                "metric": draft_metric,
                "error": draft_eval["error"] or (None if draft_exec_success else draft_exec_result.exc_type),
                "plan": draft_payload["plan"],
                "code": draft_payload["code"],
                "raw_output": draft_raw,
            },
            "feedback": {
                "structured_output_success": True,
                "review": review_payload,
                "raw_output": review_raw,
            },
            "improvement": {
                "structured_output_success": True,
                "python_valid": improved_ok,
                "execution_success": improved_exec_success,
                "submission_valid": improved_eval["submission_valid"],
                "metric": improved_metric,
                "error": improved_eval["error"] or (None if improved_exec_success else improved_exec_result.exc_type),
                "plan": improved_payload["plan"],
                "code": improved_payload["code"],
                "raw_output": improved_raw,
            },
            "metric_delta": metric_delta,
            "full_loop_completed": bool(
                draft_ok
                and draft_exec_success
                and draft_eval["submission_valid"]
                and improved_ok
                and improved_exec_success
                and improved_eval["submission_valid"]
            ),
            "runtime": {
                "draft_stdout": draft_stdout,
                "draft_stderr": draft_stderr,
                "improved_stdout": improved_stdout,
                "improved_stderr": improved_stderr,
            },
        }
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"\nSaved result: {result_file}")
        print(f"Draft metric: {draft_metric}")
        print(f"Improved metric: {improved_metric}")
        print(f"Metric delta: {metric_delta}")
        print(f"Full loop completed: {result['full_loop_completed']}")
        return result
    finally:
        logging.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Frontis-MA1-30B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--result-file", type=Path, default=ROOT / "runs" / "frontis_smoke_result.json")
    args = parser.parse_args()
    try:
        _run_one(args.model, args.base_url, args.api_key, args.result_file)
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
