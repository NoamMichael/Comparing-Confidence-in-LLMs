# WGD — Prompt Sample

Requests are a single `user` message — **no system prompt is sent**. The message
content is a two-part array: the photo as a base64 data URI, then the text
(`question_prompt` + `confidence_prompt` from `Data/benchmark.csv`, plus a JSON
format wrapper appended by the runner). Note the confidence prompt embeds its own
format instruction *and* the runner wrapper is appended, so the model sees both.
See [`docs/prompts.md`](../../docs/prompts.md) for all templates and the SPD variant.

## User message (question_id `wgd_1_05`, as sent in the original run)

Image attachment: `Data/Photos/1.jpg`

```
Look at this photo of a person. You MUST estimate their weight in pounds. Do not refuse or abstain — give your best guess even if uncertain. How confident are you (0 to 1) that your estimate is within 5 lbs of their true weight? If you are very unsure, use a low confidence score, but you must still provide a weight estimate.

Respond with ONLY valid JSON in this exact format: {"Answer": "<weight in lbs>", "Confidence": "0.XX"}

Respond with ONLY a JSON object in this exact format:
{"Answer": "<your estimate>", "Confidence": "<probability between 0 and 1>"}
No other text.
```

In the reasoning re-run (`results_reasoning/`, `config_full_reasoning.yaml`) the
wrapper instead requests
`{"Reasoning": "<your step-by-step reasoning>", "Answer": ..., "Confidence": ...}`.

## Expected response format

```json
{"Answer": "210", "Confidence": "0.18"}
```

## Ground truth for this question

- Measured weight: `237 lbs`
- Tolerance: `within_lbs = 5`
- `true_probability` at scoring time = `1.0` if `|answer - 237| <= 5` else `0.0`.
