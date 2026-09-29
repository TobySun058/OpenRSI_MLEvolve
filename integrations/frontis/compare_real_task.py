"""Run the same native MLEvolve task against two OpenAI-compatible backends."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REAL_TASK = ROOT / "integrations" / "frontis" / "real_task.py"


def run_backend(args: argparse.Namespace, model: str, base_url: str, api_key: str) -> int:
    command = [
        sys.executable,
        str(REAL_TASK),
        "--task", args.task,
        "--dataset-dir", str(args.dataset_dir),
        "--model", model,
        "--base-url", base_url,
        "--api-key", api_key,
        "--steps", str(args.steps),
        "--initial-drafts", str(args.initial_drafts),
        "--seed", str(args.seed),
        "--time-limit", str(args.time_limit),
        "--timeout", str(args.timeout),
        "--allow-prompt-tool-fallback",
    ]
    print("\n$", " ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--base-url-a", required=True)
    parser.add_argument("--api-key-a", default="EMPTY")
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--base-url-b", required=True)
    parser.add_argument("--api-key-b", default="EMPTY")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--initial-drafts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    rc = run_backend(args, args.model_a, args.base_url_a, args.api_key_a)
    if rc:
        return rc
    return run_backend(args, args.model_b, args.base_url_b, args.api_key_b)


if __name__ == "__main__":
    raise SystemExit(main())
