# WGD — Prompt Sample

## System prompt

```
You are a calibrated forecaster. For every question, commit to a single best
answer and report a confidence in [0, 1] equal to your subjective probability
that the answer is correct. Your confidence will be scored with the Brier
Score — a strictly proper scoring rule — so you minimise expected loss only by
reporting your honest belief. Do not refuse, abstain, or hedge with ranges.
Return only the JSON object requested.
```

## User prompt (question_id `wgd_1_05`)

Image attachment: `Data/Photos/1.jpg`

```
Look at this photo of a person. You MUST estimate their weight in pounds. Do not refuse or abstain — give your best guess even if uncertain. How confident are you (0 to 1) that your estimate is within 5 lbs of their true weight? If you are very unsure, use a low confidence score, but you must still provide a weight estimate.

Respond with ONLY valid JSON in this exact format: {"Answer": "<weight in lbs>", "Confidence": "0.XX"}
```

## Expected response format

```json
{"Answer": "210", "Confidence": "0.18"}
```

## Ground truth for this question

- Measured weight: `237 lbs`
- Tolerance: `within_lbs = 5`
- `true_probability` at scoring time = `1.0` if `|answer - 237| <= 5` else `0.0`.
