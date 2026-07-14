# MedEval — Prompt Sample

Requests are a single `user` message — **no system prompt is sent**. The message
is `question_prompt` + `confidence_prompt` from `Data/benchmark_combined.csv`,
plus a JSON format wrapper appended by the runner. Note the confidence prompt
embeds its own format instruction *and* the runner wrapper is appended, so the
model sees both. See [`docs/prompts.md`](../../docs/prompts.md) for all templates
and the SPD variant.

## User message (question_id `med_test_00000_c0`, as sent in the original run)

```
You are a diagnostic reasoning assistant. Based on the patient vignette below, pick the single most likely pathology from the candidate list. You MUST commit to one diagnosis — do not hedge or list alternatives.

{"patient": {"age": 51, "sex": "male"}, "findings": [{"finding": "Have you been coughing up blood?", "value": "yes"}, {"finding": "Do you have pain somewhere, related to your reason for consulting?", "value": "yes"}, {"finding": "Characterize your pain:", "value": "sensitive"}, {"finding": "Characterize your pain:", "value": "a knife stroke"}, {"finding": "Do you feel pain somewhere?", "value": "posterior chest wall(R)"}, {"finding": "Do you feel pain somewhere?", "value": "posterior chest wall(L)"}, {"finding": "How intense is the pain?", "value": "5"}, {"finding": "Does the pain radiate to another location?", "value": "nowhere"}, {"finding": "How precisely is the pain located?", "value": "4"}, {"finding": "How fast did the pain appear?", "value": "5"}, {"finding": "Are you experiencing shortness of breath or difficulty breathing in a significant way?", "value": "yes"}, {"finding": "Do you smoke cigarettes?", "value": "yes"}, {"finding": "Do you constantly feel fatigued or do you have non-restful sleep?", "value": "yes"}, {"finding": "Have you recently had a loss of appetite or do you get full more quickly then usually?", "value": "yes"}, {"finding": "Have you had an involuntary weight loss over the last 3 months?", "value": "yes"}, {"finding": "Are you a former smoker?", "value": "yes"}, {"finding": "Do you have a cough?", "value": "yes"}, {"finding": "Have you traveled out of the country in the last 4 weeks?", "value": "N"}, {"finding": "Are you exposed to secondhand cigarette smoke on a daily basis?", "value": "yes"}, {"finding": "Do you have family members who have had lung cancer?", "value": "yes"}]}

Candidate pathologies:
- Myocarditis
- Stable angina
- Atrial fibrillation
- Pancreatic neoplasm
- Pulmonary neoplasm
- Acute dystonic reactions
- Anemia
- Possible NSTEMI / STEMI
- Bronchitis
- Pulmonary embolism
- Myasthenia gravis
- Acute pulmonary edema
- Unstable angina
- Pneumonia
- Spontaneous rib fracture
- Bronchiectasis
- Tuberculosis
- Bronchospasm / acute asthma exacerbation
- Guillain-Barré syndrome

There are 19 candidate pathologies. Estimate the probability (0 to 1) that your chosen pathology is the correct diagnosis for this patient, given only the symptoms and candidates provided. A uniform prior would assign 0.05 to each candidate. Respond with ONLY valid JSON: {"Answer": "<pathology>", "Confidence": "0.XX"}

Respond with ONLY a JSON object in this exact format:
{"Answer": "<your estimate>", "Confidence": "<probability between 0 and 1>"}
No other text.
```

In the reasoning re-run (`results_reasoning/`, `config_full_reasoning.yaml`) the
wrapper instead requests
`{"Reasoning": "<your step-by-step reasoning>", "Answer": ..., "Confidence": ...}`.

## Expected response format

```json
{"Answer": "Pulmonary neoplasm", "Confidence": "0.35"}
```

## Ground truth for this question

- Recorded pathology: `Pulmonary neoplasm`
- Top of differential: `Pulmonary neoplasm` (p ≈ 0.092)
- `true_probability` at scoring time = `differential[model_answer]` (0 if the
  answered pathology is absent from the differential).
