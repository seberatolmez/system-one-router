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
