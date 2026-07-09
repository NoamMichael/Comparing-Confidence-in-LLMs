# WGD — Weight Guessing Dataset

Vision calibration benchmark. Given a photograph of a person, the model must estimate the subject's weight in pounds and report a confidence that its estimate falls within a tolerance band of the true (measured) weight.

## Task

Each question shows a photo and a tolerance `within_lbs` (1-20 lbs). The model guesses the person's weight and states how confident it is that its guess is within `within_lbs` of the true weight.

## Scoring

Ground truth is deterministic — the subject has a single measured weight, so the model's guess is either within tolerance or not:

```
true_probability = 1.0  if |answer - true_weight| <= within_lbs
                   0.0  otherwise
```

The Brier Score on the stated confidence measures whether the model's self-assessed "am I close?" signal tracks reality:

```
brier = (confidence - true_probability)^2
```

If a model guesses 155 lbs for someone who weighs 150 lbs and the tolerance is 10 lbs, then `true_probability = 1.0` (the guess is within range). If the model reported confidence 0.7, the Brier Score is `(0.7 - 1.0)^2 = 0.09`. A perfectly calibrated model would have said 1.0.

Varying `within_lbs` from 1 to 20 probes calibration at many difficulty levels from the same photo — tight tolerances are harder to be within, so a calibrated model should report lower confidence for smaller tolerances.

## Benchmark

Questions = number of labeled photos x 20 tolerance levels.

### Build

```bash
python build_benchmark.py
```

Reads `Data/labels.csv` (measured weights) and `Data/Photos/` (one `<Participant Number>.jpg` per subject), writes `Data/benchmark.csv`.

Columns: `question_id, question_prompt, confidence_prompt, photo, within_lbs, true_weight`.

### Score

Given a results CSV with `question_id, Answer, Confidence` (where `Answer` is a numeric weight in pounds):

```python
from analysis.scoring import score_wgd, murphy_decomposition
scored = score_wgd(results_df, benchmark_df)
print(f"Brier Score: {scored['brier'].mean():.4f}")
print(murphy_decomposition(scored))
```

The Murphy decomposition breaks Brier Score into:
- **Reliability** — how close confidence is to observed frequency (lower = better calibrated)
- **Resolution** — ability to discriminate between easy and hard questions (higher = better)
- **Uncertainty** — irreducible difficulty of the question set

## Expected model response format

```json
{"Answer": "155", "Confidence": "0.62"}
```
