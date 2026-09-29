# Frontis integration

This directory contains the validation harness added to test **Frontis/OpenRSI models as MLEvolve backends**.

The upstream MLEvolve search/agent implementation remains in its original top-level packages. The actual integration touches two framework files:

- `llm/model_profiles.py` — Frontis model profile and capability flags
- `llm/openai.py` — structured-output fallback and tolerant OpenAI-compatible response handling

Everything in this directory is support code for validating that integration.

## Files

| File | Purpose |
| --- | --- |
| `endpoint_probe.py` | Directly probe chat, thinking, and prompt-only JSON behavior |
| `smoke_test.py` | Small structured plan/code/execute/review compatibility test |
| `native_mini_loop.py` | Tiny native MLEvolve loop with generated toy data |
| `real_task.py` | Run one prepared MLE-Bench task through native MLEvolve |
| `compare_backends.py` | Compare two backends on the structured smoke test |
| `compare_real_task.py` | Run the same native task against two configured backends |
| `mock_openai_server.py` | Local deterministic server for interface testing |

## Recommended validation order

```bash
python integrations/frontis/endpoint_probe.py --model Frontis-MA1-30B --base-url http://127.0.0.1:8000/v1
python integrations/frontis/smoke_test.py --model Frontis-MA1-30B --base-url http://127.0.0.1:8000/v1
python integrations/frontis/native_mini_loop.py --model Frontis-MA1-30B --base-url http://127.0.0.1:8000/v1
```

Only move to a full MLE-Bench task after those interface-level tests succeed.

## Why this is separated

The original repository structure makes it easy to confuse upstream MLEvolve code with integration work. Keeping the Frontis-specific harness here makes the contribution boundary explicit while leaving the upstream architecture recognizable.
