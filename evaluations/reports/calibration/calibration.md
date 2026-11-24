# Policy calibration

Calibrated on a train split and scored once on a held-out test split. A policy wins only if it cuts cost per successful task by at least 10% and its quality difference to always-balanced has a 95% lower bound of at least -0.05.

Settings: `{"margin": 0.05, "min_saving": 0.1, "seed": 20261122, "test_fraction": 0.35, "test_size": 105, "train_margin": 0.03, "train_size": 195}`

| Decision model | Policy (fast below / confidence gate) | Test fast share | Quality diff (95% CI) | Cost per success change (95% CI) | Wins |
| --- | --- | --- | --- | --- | --- |
| `typesafe/jev-1.14` | always-balanced (nothing beat it on train) | 0% | +0.000 (+0.000 to +0.000) | +1.0% (+0.7% to +1.5%) | no |
| `google/gemini-decision-1` | always-balanced (nothing beat it on train) | 0% | +0.000 (+0.000 to +0.000) | +1.7% (+1.2% to +2.6%) | no |
| `microsoft/microsoft-decision-1.1` | always-balanced (nothing beat it on train) | 0% | +0.000 (+0.000 to +0.000) | +0.5% (+0.4% to +0.8%) | no |
| `openai/gpt-6-luna-decisions` | always-balanced (nothing beat it on train) | 0% | +0.000 (+0.000 to +0.000) | +2.2% (+1.6% to +3.4%) | no |

Items used per model (train / test / excluded):

- `typesafe/jev-1.14`: 128 / 71 / 101
- `google/gemini-decision-1`: 128 / 71 / 101
- `microsoft/microsoft-decision-1.1`: 128 / 71 / 101
- `openai/gpt-6-luna-decisions`: 128 / 71 / 101
