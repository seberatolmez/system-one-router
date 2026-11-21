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
`system-one/balanced`, `system-one/reasoning`, and `heuristic` (a keyword and
prompt-length baseline with no model call). The runner records routing
accuracy against optional `expected_tier` labels, measured cost, latency,
error rate, escalation rate, quality pass rate, and cost per successful task.
Missing token usage or pricing is left unmeasured rather than reported as zero.

### Quality scoring

Answers are graded pass/fail. Items with a `reference` answer (a number or an
option letter) are matched exactly; other items are graded by the chat model
given with `--judge-model`, and stay ungraded without one. A request that
errored counts as failed. Responses are discarded after grading, so reports
hold verdicts only. The judge model is not free: its calls are not included in
a strategy's cost.

```text
uv run python -m system_one.evaluation.cli --dataset evaluations/datasets/routing-benchmark.jsonl --strategy system-one/reasoning --strategy system-one/fast --strategy heuristic --strategy system-one/auto --judge-model <chat-model-slug>
```

`evaluations/datasets/routing-benchmark.jsonl` ships with provisional
`expected_tier` labels derived from difficulty (see
`evaluations/datasets/README.md`). To replace them with measured labels (the
cheapest tier whose answer passes grading), run all three tiers once:

```text
uv run python -m system_one.evaluation.oracle --dataset evaluations/datasets/routing-benchmark.jsonl --output evaluations/datasets/routing-benchmark.measured.jsonl --judge-model <chat-model-slug>
```

An item stays unlabeled when no tier passes or a cheaper tier could not be graded.

### Comparing decision models

`--decision-model` benchmarks decision models on routing decisions only (decide,
then policy; no completion is requested, so a run costs cents). Any model that
speaks the OpenRouter Decisions API works, and it cannot be combined with
`--strategy`:

```text
uv run python -m system_one.evaluation.cli --dataset evaluations/datasets/sample.jsonl --decision-model typesafe/jev-1.14 --decision-model google/gemini-decision-1 --output-dir evaluations/reports
```

The report adds `under_routing_rate` (routed below the expected tier),
`brier_score` (confidence calibration against tier correctness, lower is better),
and `decision_cost_per_1000_requests`. A fallback decision counts as an error and
is excluded from accuracy and calibration.

## Benchmarks

Measured on 2026-11-21. Reports: `evaluations/reports/`. Spend for every run
below, including re-runs: about $3.60 on OpenRouter.

### Setup

| | |
| --- | --- |
| Tier models | fast `google/gemini-2.5-flash`, balanced `openai/gpt-6.1-terra`, reasoning `anthropic/claude-sonnet-5.5` (`registry/models.yaml`) |
| Decision models | `typesafe/jev-1.14`, `google/gemini-decision-1`, `microsoft/microsoft-decision-1.1`, `openai/gpt-6-luna-decisions` |
| Judge | `google/gemini-4-carbon`, pass/fail; exact match where a reference answer exists |
| Data | `routing-benchmark-60.jsonl`: the first 4 items of each task-type x difficulty cell of the 300-item set |
| Labels | measured: the cheapest tier whose answer passed (57 of 60 labeled; 3 passed on no tier) |

Each tier model answered all 60 prompts once and every answer was graded once.
Routers were then scored by **replay**: a router's outcome on an item is the
measured outcome of the tier it picked, plus its own decision latency and cost.
All routers are compared on identical answers; only the routing choice differs.

### Result 1: no router beats always-balanced on this model pool

Default policy (`policies/balanced.yaml`):

| Strategy | Quality pass rate (95% CI) | Cost / successful task | Routing accuracy | Under-routing | p50 / p95 latency |
| --- | --- | --- | --- | --- | --- |
| always-fast | 0.46 (0.34–0.58) | $0.0052 | 0.47 | 0.53 | 2.4 s / 19.2 s |
| **always-balanced** | **0.95 (0.86–0.98)** | **$0.0030** | 0.53 | 0.00 | 4.0 s / 29.9 s |
| always-strong | 0.87 (0.76–0.93) | $0.0185 | 0.00 | 0.00 | 4.5 s / 47.7 s |
| heuristic | 0.70 (0.57–0.80) | $0.0042 | 0.70 | 0.28 | 3.0 s / 29.9 s |
| `system-one/auto` + Jev 1.14 | 0.95 (0.86–0.98) | $0.0031 | 0.65 | 0.00 | 4.8 s / 30.7 s |
| `system-one/auto` + Gemini Decision 1 | 0.95 (0.86–0.98) | $0.0031 | 0.53 | 0.00 | 4.8 s / 30.7 s |
| `system-one/auto` + Microsoft Decision 1.1 | 0.95 (0.86–0.98) | $0.0031 | 0.53 | 0.00 | 4.8 s / 30.8 s |
| `system-one/auto` + GPT-6 Luna Decisions | 0.95 (0.86–0.98) | $0.0032 | 0.68 | 0.00 | 4.8 s / 30.8 s |

Why:

- **The balanced model is the best model here.** No item's cheapest passing
  tier was `reasoning`, and the reasoning model passed fewer items than the
  balanced one (0.87 vs 0.95) at six times the cost per success.
- **The fast model is not cheaper per request.** Its answers are long, so it
  cost $0.0025 per request against $0.0029 for balanced, while passing 0.46.
  Price per token is a poor proxy for price per request.
- **The default policy gates almost everything to the fallback tier.** Its
  0.85 confidence threshold sends 66–99% of decisions (across the 300-item run)
  to `balanced`, and no decision model ever selected `reasoning`. So
  `system-one/auto` currently behaves like always-balanced plus ~0.8 s of
  decision latency.

### Result 2: decision model leaderboard

To show the decision models' own routing signal, the same recorded decisions are
replayed under `evaluations/policies/ungated.yaml` (the same complexity ranges,
no confidence gate). The policy was fixed before looking at results and is an
evaluation view, not a recommended runtime policy.

| Decision model | Quality pass rate (95% CI) | Routing accuracy | Under-routing | Brier score (lower is better) | Decision cost / 1,000 | Decision p50 / p95 (300 items) |
| --- | --- | --- | --- | --- | --- | --- |
| `google/gemini-decision-1` | **0.90** (0.80–0.95) | **0.77** | **0.05** | 0.267 | $0.056 | 0.80 s / 0.91 s |
| `microsoft/microsoft-decision-1.1` | 0.83 (0.72–0.91) | 0.68 | 0.14 | 0.242 | **$0.017** | 0.88 s / 1.26 s |
| `typesafe/jev-1.14` | 0.80 (0.68–0.88) | 0.70 | 0.18 | 0.211 | $0.032 | **0.75 s / 0.82 s** |
| `openai/gpt-6-luna-decisions` | 0.73 (0.61–0.83) | 0.67 | 0.25 | **0.197** | $0.072 | 0.76 s / 0.94 s |

- Gemini Decision 1 routes most accurately and under-routes least; Luna is the
  best calibrated but sends too much to the fast tier. With 60 items the
  intervals overlap, so only Gemini vs Luna is a clear gap.
- Decisions are fast and cheap: under a second at p50 and under $0.08 per
  1,000 for all four models.
- Microsoft Decision 1.1 returned 19 errors (15 non-200 responses, 4 timeouts)
  in a first 300-item run and none in the reported re-run; the other models had
  no errors in either.
- Difficulty labels are a weak proxy for the right tier: the fast model solved
  7 of 20 "medium" and 3 of 20 "hard" items. Do not score routers against
  `difficulty`.

### Reproduce

```text
# 1. Decisions (300 items, cents)
uv run python -m system_one.evaluation.cli --dataset evaluations/datasets/routing-benchmark.jsonl --output-dir evaluations/reports/decisions --decision-model typesafe/jev-1.14 --decision-model google/gemini-decision-1 --decision-model microsoft/microsoft-decision-1.1 --decision-model openai/gpt-6-luna-decisions

# 2. Tier answers, graded (60 items, about $3)
OPENROUTER_TIMEOUT_SECONDS=300 uv run python -m system_one.evaluation.cli --dataset evaluations/datasets/routing-benchmark-60.jsonl --output-dir evaluations/reports/tiers --strategy system-one/fast --strategy system-one/balanced --strategy system-one/reasoning --judge-model google/gemini-4-carbon

# 3. Replay (offline, free); add --policy evaluations/policies/ungated.yaml for the leaderboard view
uv run python -m system_one.evaluation.replay --dataset evaluations/datasets/routing-benchmark-60.jsonl --tier-report evaluations/reports/tiers/<run>.json --decision-report evaluations/reports/decisions/<run>.json --output-dir evaluations/reports/replay-default
```

### Limitations

- **Small sample.** 60 graded items; quality intervals are about ±0.1. One run
  per model, and LLM output varies between runs.
- **One judge.** `gemini-4-carbon` can be wrong and may favor some styles; only
  8 of the 60 items have exact-match references.
- **Replay, not live.** Router latency is the decision latency plus the chosen
  tier's measured latency, not a live end-to-end call. Answers are identical
  across routers by construction.
- **Re-runs.** 17 fast-tier calls failed with transport errors (most within
  200 ms) during local network outages and were re-run with the same model and grader (noted in the tier
  report). No other outcome was re-run.
- **One model pool.** Conclusions are about this registry. A pool where the
  strong tier is clearly better and the fast tier clearly cheaper per request
  could reward routing very differently.
- The decision summary in `evaluations/reports/decisions/summary.md` scores
  routing against the provisional difficulty labels; use the replay reports for
  measured accuracy.
- OpenRouter labels the Decisions API alpha; prices are list prices.

### Add your decision model

1. Confirm it appears in `GET https://openrouter.ai/api/v1/models?output_modalities=decisions`
   and accepts `choice` and `score` questions.
2. Run step 1 with your slug as an extra `--decision-model`, then step 3 with the
   committed tier report: no new completions are needed, so it costs cents.
3. Open a pull request with the decision report and the updated table.
