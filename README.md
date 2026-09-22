# System One Router

An OpenRouter-compatible, self-hosted LLM routing gateway powered by Jev for
fast decisions, deterministic policies, and measurable model selection.

## Phase 0

The repository bootstrap currently exposes liveness and readiness endpoints.
OpenRouter passthrough and routing are introduced in later milestones.

Install dependencies with `uv` and run the checks:

```text
uv sync
uv run pytest
uv run uvicorn system_one.main:app --reload
```

The service exposes:

```text
GET http://localhost:8000/health
GET http://localhost:8000/ready
```
