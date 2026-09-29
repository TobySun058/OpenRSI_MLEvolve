# Integration Notes

## Objective

Evaluate whether a Frontis/OpenRSI-released model can act as an effective backend for MLEvolve through an OpenAI-compatible API.

## Compatibility boundary

The experiment deliberately keeps MLEvolve's higher-level search and agent logic unchanged where possible. Changes are concentrated around model selection, request construction, response parsing, and platform compatibility.

## Failure modes to distinguish

When testing a new backend, separate:

- endpoint/authentication failures;
- unsupported structured-output features;
- malformed JSON or schema mismatch;
- model-level instruction-following failures;
- framework-level search/evaluation failures;
- platform-specific runtime assumptions.

The smoke scripts exist to isolate these layers before running a full MLEvolve task.

## Evaluation philosophy

A successful endpoint response is not sufficient evidence that the backend is useful for MLEvolve. Useful evaluation should eventually include tasks where model quality, structured-output reliability, and downstream search behavior can be compared across backends under the same framework settings.
