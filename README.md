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

**Status: no results are published yet.** The benchmark code, dataset and
commands are in place; numbers are added here only after a run is measured and
its reports are committed to `evaluations/reports/`.

### Decision model leaderboard

Which decision model routes best? Every model below speaks the same OpenRouter
Decisions API, so one command compares them on the 300-item
`routing-benchmark.jsonl` (about cents in API spend):

```text
uv run python -m system_one.evaluation.cli --dataset evaluations/datasets/routing-benchmark.jsonl --output-dir evaluations/reports --decision-model typesafe/jev-1.14 --decision-model typesafe/jev-1.13 --decision-model google/gemini-decision-1 --decision-model microsoft/microsoft-decision-1.1 --decision-model microsoft/microsoft-decision-1 --decision-model openai/gpt-6-luna-decisions --decision-model liquid/d1.2 --decision-model liquid/d1.2-ultra --decision-model cloudflare/clef-flash --decision-model perplexity/pplx-decider-v1.3:free
```

| Model | Routing accuracy | Under-routing | Over-routing | Brier score | Cost / 1,000 decisions | p50 / p95 latency | Error rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| _not yet measured_ | | | | | | | |

### End-to-end strategies

Compares Always Strong (`system-one/reasoning`), Always Fast
(`system-one/fast`), the `heuristic` baseline and `system-one/auto` on quality
pass rate, cost per successful task, latency and routing accuracy. See
[Quality scoring](#quality-scoring) for the command.

| Strategy | Quality pass rate | Cost / successful task | p50 / p95 latency | Routing accuracy |
| --- | --- | --- | --- | --- |
| _not yet measured_ | | | | |

### Method

- Every strategy sees the same 300 prompts, sequentially, through the same
  policy (`policies/balanced.yaml`) and model registry.
- Quality is exact match against a reference where one exists (40 public
  items) and a judge model's pass/fail otherwise.
- Routing accuracy is scored against `expected_tier`. The shipped labels are
  provisional (derived from difficulty); the `oracle` command replaces them with
  measured ones. Publish accuracy only against measured labels.
- A published run states the date, the model slugs and versions, the judge
  model, and links its JSON report.

### Limitations

- A single run per model: LLM output varies between runs, and no confidence
  intervals are computed yet.
- The judge model can be wrong and may favor answers that resemble its own.
- The dataset is 300 prompts across five task types; it says nothing about
  other domains, languages beyond the translation items, or multi-turn chats.
- Decision accuracy depends on the policy thresholds and on the 0.85 confidence
  threshold, which sends low-confidence decisions to the fallback tier. Compare
  models under one policy, not across policies.
- OpenRouter labels the Decisions API alpha; results may shift when it changes.
- Prices are those in `registry/models.yaml` and the cost OpenRouter reports for
  each decision, not your negotiated rates.

### Add your decision model

1. Confirm it appears in `GET https://openrouter.ai/api/v1/models?output_modalities=decisions`
   and accepts `choice` and `score` questions.
2. Run the leaderboard command with your slug as an extra `--decision-model`.
3. Open a pull request with the generated JSON report and the updated table. No
   code changes are needed.
