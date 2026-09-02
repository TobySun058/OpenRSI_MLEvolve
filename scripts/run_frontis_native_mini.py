"""Run a tiny native MLEvolve mini loop and persist artifacts under runs/native."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs" / "native"


def _build_task(base_dir: Path) -> tuple[Path, Path]:
    data_dir = base_dir / "task_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    desc_file = data_dir / "description.md"
    desc_file.write_text(
        "# Task goal\n"
        "Build a tiny one-feature linear-regression solution.\n\n"
        "# Evaluation\n"
        "Evaluation Metric: Mean Squared Error (MSE). Lower values are better.\n"
        "train.csv has columns `id`, `x`, `y`; test.csv has columns `id`, `x`.\n"
        "Write `submission/submission.csv` with the original test `id` values and `prediction`.\n"
        "Print `Final Validation Score: <score>` using a task-faithful MSE.\n",
        encoding="utf-8",
    )

    with (data_dir / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y"])
        writer.writerows([(1, 0, 1.0), (2, 1, 4.0), (3, 2, 7.0), (4, 3, 10.0)])

    with (data_dir / "test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x"])
        writer.writerows([(5, 4), (6, 5)])

    with (data_dir / "sample_submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prediction"])
        writer.writerows([(5, 0.0), (6, 0.0)])

    return data_dir, desc_file


def _safe_model_tag(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_journal(log_dir: Path) -> dict[str, Any] | None:
    journal_path = log_dir / "journal.json"
    if not journal_path.exists():
        return None
    return _load_json(journal_path)


def _summarize_journal(log_dir: Path, model: str, base_url: str, fallback_used: bool) -> dict[str, Any]:
    journal = _read_journal(log_dir)
    if journal is None:
        return {
            "model": model,
            "base_url": base_url,
            "prompt_tool_fallback_used": fallback_used,
            "draft_succeeded": False,
            "draft_executed": False,
            "result_parsing_succeeded": False,
            "second_agent_invoked": False,
            "second_candidate_generated": False,
            "second_candidate_executed": False,
            "full_native_mini_loop_completed": False,
            "candidates": [],
        }

    nodes = journal.get("nodes", [])
    non_root = [node for node in nodes if node.get("stage") != "root"]
    candidates = []
    for idx, node in enumerate(non_root, start=1):
        metric = ((node.get("metric") or {}).get("value"))
        parent_id = journal.get("node2parent", {}).get(node.get("id"))
        candidates.append(
            {
                "index": idx,
                "node_id": node.get("id"),
                "parent_id": parent_id,
                "stage": node.get("stage"),
                "execution_success": node.get("exc_type") is None and node.get("_term_out") is not None,
                "exc_type": node.get("exc_type"),
                "metric": metric,
                "is_buggy": node.get("is_buggy"),
                "is_valid": node.get("is_valid"),
                "analysis": node.get("analysis"),
            }
        )

    draft = next((node for node in candidates if node["stage"] == "draft"), None)
    second = candidates[1] if len(candidates) > 1 else None

    return {
        "model": model,
        "base_url": base_url,
        "prompt_tool_fallback_used": fallback_used,
        "draft_succeeded": draft is not None,
        "draft_executed": bool(draft and draft["execution_success"]),
        "result_parsing_succeeded": bool(draft and draft["analysis"]),
        "second_agent_invoked": second is not None,
        "second_candidate_generated": second is not None,
        "second_candidate_executed": bool(second and second["execution_success"]),
        "full_native_mini_loop_completed": bool(
            draft
            and draft["execution_success"]
            and second
            and second["execution_success"]
        ),
        "candidates": candidates,
    }


def _write_metadata(run_dir: Path, args: argparse.Namespace, run_cmd: list[str]) -> None:
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "seed": args.seed,
        "steps": args.steps,
        "initial_drafts": args.initial_drafts,
        "serving_backend": "Ollama" if "11434" in args.base_url else "OpenAI-compatible",
        "quantization": args.quantization,
        "prompt_tool_fallback_used": bool(args.allow_prompt_tool_fallback),
        "python_version": sys.version,
        "platform": platform.platform(),
        "command": run_cmd,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Frontis-MA1-30B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--initial-drafts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--allow-prompt-tool-fallback", action="store_true")
    parser.add_argument("--quantization", default="Q4_K_M")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{_safe_model_tag(args.model)}_{timestamp}"
    run_dir = RUNS_ROOT / run_name
    task_root = run_dir / "task"
    data_dir, desc_file = _build_task(task_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "run.py"),
        f"data_dir={data_dir}",
        f"desc_file={desc_file}",
        "eval=null",
        "workspace_dir=" + str(run_dir / "workspace_root"),
        "log_dir=" + str(run_dir / "workspace_root"),
        "exp_name=native_mini",
        "exp_id=native_mini",
        f"agent.steps={args.steps}",
        f"agent.initial_drafts={args.initial_drafts}",
        "agent.time_limit=600",
        f"agent.seed={args.seed}",
        "agent.data_preview=True",
        "agent.use_diff_mode=False",
        "agent.use_stepwise_generation=False",
        "agent.use_evolution=False",
        "agent.use_fusion=False",
        "agent.use_aggregation=False",
        "agent.use_global_memory=False",
        "agent.check_data_leakage=False",
        "agent.code.model=" + args.model,
        "agent.feedback.model=" + args.model,
        "agent.code.base_url=" + args.base_url,
        "agent.feedback.base_url=" + args.base_url,
        "agent.code.api_key=" + args.api_key,
        "agent.feedback.api_key=" + args.api_key,
        "agent.search.parallel_search_num=1",
        "agent.search.num_drafts=1",
        "agent.search.num_bugs=1",
        "agent.search.num_improves=1",
        "agent.search.top_candidates_size=1",
        "cpu_number=1",
        "start_cpu_id=0",
        f"exec.timeout={args.timeout}",
        "preprocess_data=False",
        "copy_data=True",
        "use_grading_server=False",
        "coldstart.use_coldstart=False",
    ]

    env = os.environ.copy()
    if args.allow_prompt_tool_fallback:
        env["MLEVOLVE_ALLOW_PROMPT_TOOL_FALLBACK"] = "1"

    _write_metadata(run_dir, args, cmd)
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
    )

    (run_dir / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (run_dir / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")

    workspace_root = run_dir / "workspace_root"
    log_dirs = sorted(workspace_root.glob("*/logs"))
    latest_log_dir = log_dirs[-1] if log_dirs else None
    if latest_log_dir is not None:
        target_logs = run_dir / "logs"
        if target_logs.exists():
            shutil.rmtree(target_logs)
        shutil.copytree(latest_log_dir, target_logs)

    summary = _summarize_journal(run_dir / "logs", args.model, args.base_url, bool(args.allow_prompt_tool_fallback))
    summary["returncode"] = completed.returncode
    summary["run_dir"] = str(run_dir)
    summary["stdout_path"] = str(run_dir / "stdout.txt")
    summary["stderr_path"] = str(run_dir / "stderr.txt")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Saved run to: {run_dir}")
    print(f"Return code: {completed.returncode}")
    print(f"Full native mini loop completed: {summary['full_native_mini_loop_completed']}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
