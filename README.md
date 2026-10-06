# System One Router

An OpenRouter-compatible, self-hosted LLM routing gateway powered by Jev for
fast decisions, deterministic policies, and measurable model selection.

## Current Status

The bootstrap, OpenRouter passthrough, virtual-model routing, and SSE streaming
are implemented.

Install dependencies with `uv` and run the checks:

```text
uv sync
uv run pytest
uv run uvicorn system_one.main:app --reload
```

For local passthrough, copy `.env.example` to `.env`, set
`OPENROUTER_API_KEY`, and keep the example gateway key or replace it:

```text
SYSTEM_ONE_API_KEY=local-dev-key
OPENROUTER_API_KEY=your-openrouter-key
```

The service exposes health endpoints:

```text
GET http://localhost:8000/health
GET http://localhost:8000/ready
```

The OpenRouter-compatible endpoints are:

```text
GET  http://localhost:8000/api/v1/models
POST http://localhost:8000/api/v1/chat/completions
```

The gateway accepts explicit OpenRouter model IDs such as `openai/gpt-4o-mini`
passed through to OpenRouter, and virtual routing models `system-one/auto`,
`system-one/fast`, `system-one/balanced`, and `system-one/reasoning`. Virtual
models resolve through the Jev decision engine, the deterministic policy engine,
and the model registry (`registry/models.yaml`) to a concrete OpenRouter model;
`system-one/fast|balanced|reasoning` map directly to their tier without a
decision-engine call. Requests with `stream: true` receive the provider's SSE
events, including any final usage chunk the provider sends.

Run the example request after starting the service:

```text
sh examples/curl/chat-completion.sh
```

## Evaluation framework

The evaluation runner applies the same JSONL dataset to each selected virtual
model strategy and writes a machine-readable run plus a Markdown comparison.
The included `evaluations/datasets/sample.jsonl` is a small smoke-test dataset,
not a measured benchmark or a quality claim. Live inference requires an
`OPENROUTER_API_KEY`; Jev decisions are enabled separately with the Jev
settings. The configured policy and registry files are used by the runner.

```text
uv run python -m system_one.evaluation.cli --dataset evaluations/datasets/sample.jsonl --strategy system-one/auto --strategy system-one/fast --output-dir evaluations/reports
```

Available strategies are `system-one/auto`, `system-one/fast`,
`system-one/balanced`, and `system-one/reasoning`. The runner records routing
accuracy against optional `expected_tier` labels, measured cost, latency,
error rate, and escalation rate. Missing token usage or pricing is left
unmeasured rather than reported as zero. Quality scoring and benchmark claims
are intentionally not included in this framework milestone.
