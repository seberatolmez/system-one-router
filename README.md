# System One Router

An OpenRouter-compatible, self-hosted LLM routing gateway powered by Jev for
fast decisions, deterministic policies, and measurable model selection.

## Current Status

Phase 0 bootstrap and the initial OpenRouter passthrough are implemented.
Automatic routing, streaming, and virtual `system-one/*` models are not enabled
yet.

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

Milestone 1 accepts explicit OpenRouter model IDs such as
`openai/gpt-4o-mini`. Requests using `system-one/auto`,
`system-one/fast`, `system-one/balanced`, or `system-one/reasoning` are rejected
until automatic routing is implemented. Streaming requests are also deferred.

Run the example request after starting the service:

```text
sh examples/curl/chat-completion.sh
```
