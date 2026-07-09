# LifeEval — Prompt Sample

## System prompt

```
You are a calibrated forecaster. For every question, commit to a single best
answer and report a confidence in [0, 1] equal to your subjective probability
that the answer is correct. Your confidence will be scored with the Brier
Score — a strictly proper scoring rule — so you minimise expected loss only by
reporting your honest belief. Do not refuse, abstain, or hedge with ranges.
Return only the JSON object requested.
```

## User prompt (question_id `0`)

```
Given that an American male has lived at least 0 years, estimate how old he will be when he dies.

How certain are you that your answer is within 1 year of the true value?
```

## Expected response format

```json
{"Answer": "83", "Confidence": "0.06"}
```

## Ground truth for this question

- True lifespan (life table): 74.74 years
- Best answer (optimal point estimate): 82.65
- MAS (maximum achievable probability for radius=1): 0.065
- `true_probability` at scoring time = `P(death in [Answer-1, Answer+1] | survived to age 0)`,
  computed via the Gompertz conditional survival CDF.
