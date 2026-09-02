"""Probe the OpenAI-compatible behaviors used by MLEvolve.

This intentionally talks to the endpoint directly rather than importing the
full search stack, so interface failures are easy to distinguish from agent
or execution failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from openai import OpenAI


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Frontis-MA1-30B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def _client(args: argparse.Namespace) -> OpenAI:
    return OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=120)


def _text_completion(client: OpenAI, args: argparse.Namespace, **kwargs: Any):
    return client.chat.completions.create(
        model=args.model,
        max_tokens=args.max_tokens,
        **kwargs,
    )


def main() -> int:
    args = _args()
    client = _client(args)
    failures: list[str] = []

    print(f"Endpoint: {args.base_url}")
    print(f"Model: {args.model}")

    try:
        response = _text_completion(
            client,
            args,
            messages=[{"role": "user", "content": "Reply with exactly: FRONTIS_OK"}],
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""
        print(f"[PASS] chat completion: {text.strip()!r}")
    except Exception as exc:
        failures.append(f"chat completion: {exc}")
        print(f"[FAIL] chat completion: {exc}")

    try:
        # This matches MLEvolve's Frontis/Qwen path: thinking stays enabled and
        # JSON is enforced by the prompt/parser rather than response_format.
        response = _text_completion(
            client,
            args,
            messages=[
                {"role": "system", "content": "Return only one JSON object."},
                {"role": "user", "content": '{"ok": true, "value": 7}'},
            ],
            temperature=0.6,
            extra_body={"enable_thinking": True},
        )
        message = response.choices[0].message
        payload = json.loads(message.content or "")
        if payload.get("ok") is not True:
            raise ValueError(f"unexpected JSON payload: {payload!r}")
        reasoning = getattr(message, "reasoning_content", None)
        print(f"[PASS] thinking + prompt-only JSON: reasoning_content={bool(reasoning)}")
    except Exception as exc:
        failures.append(f"thinking + prompt-only JSON: {exc}")
        print(f"[FAIL] thinking + prompt-only JSON: {exc}")

    tool = {
        "type": "function",
        "function": {
            "name": "submit_probe",
            "description": "Submit the probe result.",
            "parameters": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        },
    }
    try:
        response = _text_completion(
            client,
            args,
            messages=[{"role": "user", "content": "Call submit_probe with ok=true."}],
            temperature=0.7,
            extra_body={"enable_thinking": False},
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "submit_probe"}},
        )
        calls = response.choices[0].message.tool_calls or []
        if not calls or calls[0].function.name != "submit_probe":
            raise ValueError(f"expected submit_probe tool call, got {calls!r}")
        arguments = json.loads(calls[0].function.arguments or "{}")
        if arguments.get("ok") is not True:
            raise ValueError(f"unexpected tool arguments: {arguments!r}")
        print("[PASS] required function/tool call")
    except Exception as exc:
        failures.append(f"required function/tool call: {exc}")
        print(f"[FAIL] required function/tool call: {exc}")

    if failures:
        print("\nCompatibility test failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nAll endpoint compatibility checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
