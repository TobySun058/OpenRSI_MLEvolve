# Frontis / OpenRSI Integration for MLEvolve

An experimental integration of **Frontis/OpenRSI-released language models** into the open-source **MLEvolve** machine-learning engineering framework.

This repository is based on [InternScience/MLEvolve](https://github.com/InternScience/MLEvolve). The work here focuses on one concrete question:

> Can a Frontis/OpenRSI model be used as an MLEvolve backend with reliable structured outputs and minimal changes to the upstream search pipeline?

## Scope of this repository

The underlying MLEvolve search engine, agents, memory, and evaluation framework come from the upstream project. My changes focus on the **model-integration layer and compatibility tooling**, including:

- adding a Frontis-compatible model profile;
- handling OpenAI-compatible endpoints;
- adding prompt-based structured-output fallback when native JSON-schema behavior is unavailable;
- normalizing structured response extraction;
- making CPU-affinity behavior safer on non-Linux / unsupported platforms;
- adding endpoint, smoke-test, and native mini-loop scripts;
- adding comparison scripts for Frontis and other compatible model backends.

The goal is not to reimplement MLEvolve, but to test how easily a newly released AI4AI model can be plugged into an existing autonomous MLE system.

## Upstream projects

- **MLEvolve:** https://github.com/InternScience/MLEvolve
- **OpenRSI / Frontis:** https://github.com/FrontisAI/OpenRSI

Please refer to those projects for the original frameworks, papers, model releases, setup requirements, and licenses.

## Integration map

```text
Frontis / OpenRSI model
        │
        │ OpenAI-compatible endpoint
        ▼
MLEvolve model profile
        │
        ├── structured-output request
        │
        └── prompt-based JSON fallback
        ▼
MLEvolve agents
        │
        ▼
search / execution / evaluation loop
```

## Relevant files

```text
llm/
  model_profiles.py              model-specific behavior and profiles
  openai.py                      OpenAI-compatible backend path

scripts/
  test_frontis_endpoint.py       endpoint connectivity / response check
  run_frontis_smoke.py           focused Frontis smoke test
  run_frontis_native_mini.py     small native MLEvolve loop
  compare_models.py              backend comparison helper
  mock_openai_server.py          local compatibility testing
  run_native_real_task.py        small real-task runner
```

## Setup

MLEvolve has several dependency groups. Follow the upstream installation process first:

```bash
pip install --no-deps -r requirements_base.txt
pip install --no-deps -r requirements_ml.txt
pip install --no-deps -r requirements_domain.txt
```

Then configure a compatible model endpoint in `config/config.yaml` or through the configuration path used by the test scripts.

Do **not** commit credentials. API keys and endpoint secrets should stay in local configuration or environment variables.

## Suggested validation sequence

Start with the smallest checks before running an expensive search:

```bash
python scripts/test_frontis_endpoint.py
python scripts/run_frontis_smoke.py
python scripts/run_frontis_native_mini.py
```

Then move to `scripts/run_native_real_task.py` or the standard MLEvolve entry point once the backend is producing stable structured responses.

## What I changed

The integration work is intentionally narrow and auditable. The main contribution is adapting MLEvolve's model boundary so that a Frontis/OpenRSI backend can be exercised without rewriting the rest of the agent/search framework.

Key engineering issues addressed include:

1. **Model profile support**  
   Frontis-specific behavior can be selected explicitly instead of being treated as a generic backend.

2. **Structured-output compatibility**  
   When strict structured-output features are unavailable or inconsistent, the integration can fall back to prompting for JSON and parsing the response defensively.

3. **Cross-platform execution**  
   Platform-dependent CPU-affinity behavior is guarded so basic tests can run outside the original Linux environment.

4. **Progressive validation**  
   Small endpoint and mini-loop tests make it possible to distinguish model/API issues from failures in the full MLEvolve pipeline.

## Repository status

This is an **integration experiment**, not a fork claiming ownership of the MLEvolve system. The repository preserves upstream structure so that the changes can be compared against the original implementation.

The current focus is backend compatibility and discriminative testing rather than reproducing MLEvolve leaderboard results.

## Attribution

MLEvolve is an open-source project by InternScience and collaborators. OpenRSI / Frontis models and related artifacts are released by FrontisAI and collaborators. This repository builds on those public projects and retains the upstream license file.
