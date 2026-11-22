# Benchmark summary: routing-benchmark

- Run ID: `20261122T171112663020Z`
- Created at: `2026-11-22T17:11:12.663020+00:00`
- Dataset items: 300
- Metrics use measured provider responses and configured registry prices;
  missing measurements are excluded and reported through sample counts in JSON.

| Strategy | routing_accuracy | under_routing_rate | escalation_rate | brier_score | decision_cost_per_1000_requests | p50_latency_ms | p95_latency_ms | error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| typesafe/jev-1.14 | 0.453333 fraction | 0.336667 fraction | 0.21 fraction | 0.317952 squared error | 0.0323116 cost units/1000 requests | 777.44 milliseconds | 860.568 milliseconds | 0 fraction |
| google/gemini-decision-1 | 0.343333 fraction | 0.333333 fraction | 0.323333 fraction | 0.262698 squared error | 0.0555833 cost units/1000 requests | 853.974 milliseconds | 1030.39 milliseconds | 0 fraction |
| microsoft/microsoft-decision-1.1 | 0.344482 fraction | 0.331104 fraction | 0.324415 fraction | 0.292724 squared error | 0.0166746 cost units/1000 requests | 911.802 milliseconds | 1325.99 milliseconds | 0.00333333 fraction |
| openai/gpt-6-luna-decisions | 0.453333 fraction | 0.366667 fraction | 0.18 fraction | 0.325778 squared error | 0.072093 cost units/1000 requests | 878.212 milliseconds | 1035.36 milliseconds | 0 fraction |

This sample report is an evaluation artifact, not a claim that any strategy is better.
