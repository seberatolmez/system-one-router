# Evaluation datasets

## `sample.jsonl`

Six-item smoke-test dataset. Not a benchmark.

## `routing-benchmark.jsonl`

300 prompts: 5 task types (`coding`, `summarization`, `question_answering`,
`analysis`, `translation`) x 3 difficulty levels (`easy`, `medium`, `hard`) x 20.

| Part | Items | Source | License |
| --- | --- | --- | --- |
| coding, summarization, analysis, translation, easy question answering | 260 | written for this repository, drafted with an LLM assistant | MIT |
| medium/hard question answering (GSM8K) | 20 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) test split | MIT |
| medium/hard question answering (MMLU-Pro) | 20 | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) test split | MIT |

Each record carries `source` and `license`. The 40 public items also carry a
`reference` answer (a number for GSM8K, an option letter for MMLU-Pro) for
exact-match scoring.

Public items were not sampled at random: they are the first matching rows of
each split. GSM8K items are picked by solution length (3-4 steps = medium, 6 or
more = hard); MMLU-Pro items by subject (business, economics, health,
psychology, history and other = medium; physics, chemistry, engineering and
law = hard). Difficulty here is a prior, not a measurement.

**`expected_tier` is provisional.** It is derived from `difficulty`
(`easy`=fast, `medium`=balanced, `hard`=reasoning) and is a prior, not a
measurement. Do not read routing accuracy against these labels as a quality
result until the labels are replaced with measured ones.

Item fields: `id`, `messages`, and optional `expected_tier`, `task_type`,
`difficulty`, `source`, `license`, `reference`.
