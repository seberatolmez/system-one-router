# Benchmark summary: routing-benchmark

- Run ID: `20261121T055154448320Z`
- Created at: `2026-11-21T05:51:54.448320+00:00`
- Dataset items: 300
- Metrics use measured provider responses and configured registry prices;
  missing measurements are excluded and reported through sample counts in JSON.

| Strategy | routing_accuracy | under_routing_rate | escalation_rate | brier_score | decision_cost_per_1000_requests | p50_latency_ms | p95_latency_ms | error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| typesafe/jev-1.14 | 0.45 fraction | 0.336667 fraction | 0.213333 fraction | 0.318857 squared error | 0.0323116 cost units/1000 requests | 752.048 milliseconds | 816.886 milliseconds | 0 fraction |
| google/gemini-decision-1 | 0.343333 fraction | 0.333333 fraction | 0.323333 fraction | 0.262698 squared error | 0.0555833 cost units/1000 requests | 796.908 milliseconds | 910.538 milliseconds | 0 fraction |
| microsoft/microsoft-decision-1.1 | 0.343333 fraction | 0.333333 fraction | 0.323333 fraction | 0.293078 squared error | 0.0166678 cost units/1000 requests | 876.308 milliseconds | 1255.96 milliseconds | 0 fraction |
| openai/gpt-6-luna-decisions | 0.453333 fraction | 0.366667 fraction | 0.18 fraction | 0.325778 squared error | 0.072093 cost units/1000 requests | 764.5 milliseconds | 940.742 milliseconds | 0 fraction |

This sample report is an evaluation artifact, not a claim that any strategy is better.
