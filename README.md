# Frontis / OpenRSI Integration for MLEvolve

An experimental integration of **Frontis/OpenRSI-released language models** into the open-source **MLEvolve** machine-learning engineering framework.

This repository is based on [InternScience/MLEvolve](https://github.com/InternScience/MLEvolve). The work here focuses on one concrete question:

> Can a Frontis/OpenRSI model serve as an MLEvolve backend through an OpenAI-compatible API while preserving the structured interactions the agent pipeline expects?

## Contribution boundary

The underlying MLEvolve search engine, agents, memory, execution, and evaluation framework are upstream work.

My integration changes are concentrated in:

- `llm/model_profiles.py` — Frontis model profile/capability handling;
- `llm/openai.py` — structured-output compatibility and tolerant JSON extraction;
- `integrations/frontis/` — endpoint probes, smoke tests, native mini-loops, backend comparison, and real-task runners;
- cross-platform guards for runtime assumptions such as CPU affinity.

This repository does **not** claim ownership of MLEvolve itself.

## Integration architecture

```text
Frontis / OpenRSI model
        │
        │ OpenAI-compatible API
        ▼
llm/model_profiles.py
        │
        ▼
llm/openai.py
   ├── native structured output when supported
   └── prompt-only JSON fallback + defensive parsing
        │
        ▼
upstream MLEvolve agents / search / execution
        │
        ▼
integration validation harness
integrations/frontis/
```

## Validation workflow

Start with the interface boundary:

```bash
python integrations/frontis/endpoint_probe.py \
  --model Frontis-MA1-30B \
  --base-url http://127.0.0.1:8000/v1
```

Then run the structured smoke test:

```bash
python integrations/frontis/smoke_test.py \
  --model Frontis-MA1-30B \
  --base-url http://127.0.0.1:8000/v1
```

Then test a small native MLEvolve loop:

```bash
python integrations/frontis/native_mini_loop.py \
  --model Frontis-MA1-30B \
  --base-url http://127.0.0.1:8000/v1
```

The real-task runner is available for an already prepared MLE-Bench dataset:

```bash
python integrations/frontis/real_task.py \
  --task <competition-id> \
  --dataset-dir <mle-bench-root> \
  --model Frontis-MA1-30B \
  --base-url http://127.0.0.1:8000/v1 \
  --allow-prompt-tool-fallback
```

See [integrations/frontis/README.md](integrations/frontis/README.md) for the full validation harness.

## Engineering issues addressed

### 1. Explicit model profiling

Frontis is handled explicitly at the model boundary instead of being silently treated as another model family based only on implementation ancestry.

### 2. Structured-output fallback

MLEvolve frequently expects schema-shaped outputs. When a backend does not reliably support the exact structured-output mechanism used by another provider, the integration can enforce JSON through prompting and recover a schema-shaped object defensively.

### 3. OpenAI-compatible endpoints

The integration preserves the existing MLEvolve call path while allowing a separately served compatible endpoint.

### 4. Cross-platform validation

Small tests isolate endpoint/model behavior before expensive native search runs, and platform-specific runtime assumptions are guarded where needed.

## Upstream projects

- **MLEvolve:** https://github.com/InternScience/MLEvolve
- **OpenRSI / Frontis:** https://github.com/FrontisAI/OpenRSI

Please refer to those projects for the original frameworks, papers, model releases, setup, and licenses.

## Status

This is an integration experiment rather than a new ML-engineering framework. The current focus is backend compatibility and discriminative testing, not reproducing or claiming upstream leaderboard results.
