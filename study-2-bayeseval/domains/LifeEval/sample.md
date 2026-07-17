# LifeEval — Prompt Sample

Requests are a single `user` message — **no system prompt is sent**. The message
is `question_prompt` + `confidence_prompt` from `Data/benchmark.csv`, plus a JSON
format wrapper appended by the runner (`src/runner/openrouter_client.py`). See
[`docs/prompts.md`](../../docs/prompts.md) for all templates and the SPD variant.

## User message (question_id `0`, as sent in the original run)

```
Given that an American male has lived at least 0 years, estimate how old he will be when he dies.

How certain are you that your answer is within 1 year of the true value?

Respond with ONLY a JSON object in this exact format:
{"Answer": "<your estimate>", "Confidence": "<probability between 0 and 1>"}
No other text.
```

In the reasoning re-run (`results_reasoning/`, `config_full_reasoning.yaml`) the
wrapper instead requests
`{"Reasoning": "<your step-by-step reasoning>", "Answer": ..., "Confidence": ...}`.

## Expected response format

```json
{"Answer": "83", "Confidence": "0.06"}
```

## Ground truth for this question

- True lifespan (life table): 74.74 years
- Best answer (optimal integer point estimate): 86
- MAS (maximum achievable probability for radius=1): 0.065
- `true_probability` at scoring time = `P(death in [Answer-1, Answer+1) | survived to age 0)`,
  read directly from the life table's per-year death probabilities (empirical rule).
