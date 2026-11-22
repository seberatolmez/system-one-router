# Benchmark summary: routing-benchmark-60-measured

- Run ID: `20261121T104810603540Z`
- Created at: `2026-11-21T10:48:10.603540+00:00`
- Dataset items: 60
- Metrics use measured provider responses and configured registry prices;
  missing measurements are excluded and reported through sample counts in JSON.

| Strategy | quality_pass_rate | cost_per_successful_task | routing_accuracy | under_routing_rate | escalation_rate | brier_score | p50_latency_ms | p95_latency_ms | error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always-fast | 0.457627 fraction | 0.00517957 cost units/passed task | 0.473684 fraction | 0.526316 fraction | 0 fraction | n/a | 2379.6 milliseconds | 19154.6 milliseconds | 0 fraction |
| always-balanced | 0.95 fraction | 0.00304265 cost units/passed task | 0.526316 fraction | 0 fraction | 0.473684 fraction | n/a | 4008.93 milliseconds | 29943.8 milliseconds | 0 fraction |
| always-strong | 0.866667 fraction | 0.0185121 cost units/passed task | 0 fraction | 0 fraction | 1 fraction | n/a | 4526.99 milliseconds | 47668.9 milliseconds | 0 fraction |
| heuristic | 0.7 fraction | 0.00424017 cost units/passed task | 0.701754 fraction | 0.280702 fraction | 0.0175439 fraction | n/a | 2988.03 milliseconds | 29943.8 milliseconds | 0 fraction |
| typesafe/jev-1.14 | 0.8 fraction | 0.00367575 cost units/passed task | 0.701754 fraction | 0.175439 fraction | 0.122807 fraction | 0.210558 squared error | 3788.51 milliseconds | 30719.4 milliseconds | 0 fraction |
| google/gemini-decision-1 | 0.9 fraction | 0.00336023 cost units/passed task | 0.77193 fraction | 0.0526316 fraction | 0.175439 fraction | 0.266538 squared error | 4707.37 milliseconds | 30727.7 milliseconds | 0 fraction |
| microsoft/microsoft-decision-1.1 | 0.833333 fraction | 0.00352147 cost units/passed task | 0.684211 fraction | 0.140351 fraction | 0.175439 fraction | 0.242182 squared error | 4728.28 milliseconds | 30803.7 milliseconds | 0 fraction |
| openai/gpt-6-luna-decisions | 0.733333 fraction | 0.00400924 cost units/passed task | 0.666667 fraction | 0.245614 fraction | 0.0877193 fraction | 0.197296 squared error | 3289.75 milliseconds | 30751.6 milliseconds | 0 fraction |

This sample report is an evaluation artifact, not a claim that any strategy is better.
