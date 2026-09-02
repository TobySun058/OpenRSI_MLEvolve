"""Run a real native MLEvolve task from an existing MLE-Bench dataset root."""

from __future__ import annotations

import argparse
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
RUNS_ROOT = ROOT / "runs" / "real_tasks"


def _safe_tag(text: str) -> str:
    return text.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_task_paths(dataset_dir: Path, task: str) -> tuple[Path, Path]:
    data_dir = dataset_dir / task / "prepared" / "public"
    desc_file = data_dir / "description.md"
    if not data_dir.exists():
        raise FileNotFoundError(f"Task data dir not found: {data_dir}")
    if not desc_file.exists():
        raise FileNotFoundError(f"Task description not found: {desc_file}")
    return data_dir, desc_file


def _read_journal(log_dir: Path) -> dict[str, Any] | None:
    journal_path = log_dir / "journal.json"
    if not journal_path.exists():
        return None
    return _load_json(journal_path)


def _summarize_journal(log_dir: Path, model: str, task: str, base_url: str, fallback_used: bool) -> dict[str, Any]:
    journal = _read_journal(log_dir)
    if journal is None:
        return {
            "task": task,
            "model": model,
            "base_url": base_url,
            "prompt_tool_fallback_used": fallback_used,
            "candidate_count": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "best_metric": None,
            "best_step": None,
            "candidates": [],
        }

    nodes = journal.get("nodes", [])
    node2parent = journal.get("node2parent", {})
    node_map = {node.get("id"): node for node in nodes}
    candidates = []
    best_metric = None
    best_step = None

    for node in nodes:
        if node.get("stage") == "root":
            continue
        metric = ((node.get("metric") or {}).get("value"))
        parent_id = node2parent.get(node.get("id"))
        parent_metric = None
        if parent_id and parent_id in node_map:
            parent_metric = ((node_map[parent_id].get("metric") or {}).get("value"))
        metric_delta = None
        if isinstance(metric, (int, float)) and isinstance(parent_metric, (int, float)):
            metric_delta = metric - parent_metric
        is_new_best = False
        if isinstance(metric, (int, float)):
            if best_metric is None or metric > best_metric:
                best_metric = metric
                best_step = node.get("step")
                is_new_best = True
        candidates.append(
            {
                "step": node.get("step"),
                "node_id": node.get("id"),
                "parent_id": parent_id,
                "stage": node.get("stage"),
                "execution_success": node.get("exc_type") is None and node.get("_term_out") is not None,
                "error": node.get("exc_type"),
                "metric": metric,
                "parent_metric": parent_metric,
                "metric_delta": metric_delta,
                "is_buggy": node.get("is_buggy"),
                "is_valid": node.get("is_valid"),
                "new_best": is_new_best,
            }
        )

    successful = sum(1 for c in candidates if c["execution_success"])
    failed = len(candidates) - successful
    return {
        "task": task,
        "model": model,
        "base_url": base_url,
        "prompt_tool_fallback_used": fallback_used,
        "candidate_count": len(candidates),
        "successful_executions": successful,
        "failed_executions": failed,
        "best_metric": best_metric,
        "best_step": best_step,
        "candidates": candidates,
    }


def _write_metadata(run_dir: Path, args: argparse.Namespace, run_cmd: list[str]) -> None:
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "task": args.task,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "steps": args.steps,
        "initial_drafts": args.initial_drafts,
        "time_limit_seconds": args.time_limit,
        "exec_timeout_seconds": args.timeout,
        "serving_backend": "Ollama" if "11434" in args.base_url else "OpenAI-compatible",
        "quantization": args.quantization,
        "prompt_tool_fallback_used": bool(args.allow_prompt_tool_fallback),
        "use_grading_server": bool(args.use_grading_server),
        "python_version": sys.version,
        "platform": platform.platform(),
        "command": run_cmd,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--initial-drafts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--allow-prompt-tool-fallback", action="store_true")
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--use-grading-server", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    data_dir, desc_file = _resolve_task_paths(dataset_dir, args.task)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{_safe_tag(args.task)}__{_safe_tag(args.model)}__{timestamp}"
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "run.py"),
        f"exp_id={args.task}",
        f"dataset_dir={dataset_dir}",
        f"data_dir={data_dir}",
        f"desc_file={desc_file}",
        "workspace_dir=" + str(run_dir / "workspace_root"),
        "log_dir=" + str(run_dir / "workspace_root"),
        f"exp_name={args.task}",
        f"agent.steps={args.steps}",
        f"agent.initial_drafts={args.initial_drafts}",
        f"agent.time_limit={args.time_limit}",
        f"agent.seed={args.seed}",
        "agent.data_preview=True",
        "agent.use_diff_mode=False",
        "agent.use_stepwise_generation=False",
        "agent.use_evolution=False",
        "agent.use_fusion=False",
        "agent.use_aggregation=False",
        "agent.use_global_memory=False",
        "agent.check_data_leakage=False",
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
        f"use_grading_server={str(bool(args.use_grading_server))}",
        "coldstart.use_coldstart=False",
        "agent.code.model=" + args.model,
        "agent.feedback.model=" + args.model,
        "agent.code.base_url=" + args.base_url,
        "agent.feedback.base_url=" + args.base_url,
        "agent.code.api_key=" + args.api_key,
        "agent.feedback.api_key=" + args.api_key,
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

    summary = _summarize_journal(run_dir / "logs", args.model, args.task, args.base_url, bool(args.allow_prompt_tool_fallback))
    summary["returncode"] = completed.returncode
    summary["run_dir"] = str(run_dir)
    summary["stdout_path"] = str(run_dir / "stdout.txt")
    summary["stderr_path"] = str(run_dir / "stderr.txt")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Saved run to: {run_dir}")
    print(f"Return code: {completed.returncode}")
    print(f"Candidates: {summary['candidate_count']}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
