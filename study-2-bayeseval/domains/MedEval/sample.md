# MedEval — Prompt Sample

## System prompt

```
You are a calibrated forecaster. For every question, commit to a single best
answer and report a confidence in [0, 1] equal to your subjective probability
that the answer is correct. Your confidence will be scored with the Brier
Score — a strictly proper scoring rule — so you minimise expected loss only by
reporting your honest belief. Do not refuse, abstain, or hedge with ranges.
Return only the JSON object requested.
```

## User prompt (question_id `med_test_00000`)

```
You are a diagnostic reasoning assistant. Based on the patient vignette below, pick the single most likely pathology from the candidate list. You MUST commit to one diagnosis — do not hedge or list alternatives.

{"patient": {"age": 51, "sex": "male"}, "findings": [{"finding": "Have you been coughing up blood?", "value": "yes"}, {"finding": "Do you have pain somewhere, related to your reason for consulting?", "value": "yes"}, {"finding": "Characterize your pain", "value": "sensitive"}, {"finding": "Characterize your pain", "value": "a knife stroke"}, {"finding": "Do you feel pain somewhere?", "value": "posterior chest wall(R)"}, {"finding": "Do you feel pain somewhere?", "value": "posterior chest wall(L)"}, {"finding": "How intense is the pain?", "value": "5"}, {"finding": "Does the pain radiate to another location?", "value": "nowhere"}, {"finding": "How precisely is the pain located?", "value": "4"}, {"finding": "How fast did the pain appear?", "value": "5"}, {"finding": "Are you experiencing shortness of breath or difficulty breathing in a significant way?", "value": "yes"}, {"finding": "Do you smoke cigarettes?", "value": "yes"}, {"finding": "Do you constantly feel fatigued or do you have non-restful sleep?", "value": "yes"}, {"finding": "Have you recently had a loss of appetite or do you get full more quickly then usually?", "value": "yes"}, {"finding": "Have you had an involuntary weight loss over the last 3 months?", "value": "yes"}, {"finding": "Are you a former smoker?", "value": "yes"}, {"finding": "Do you have a cough?", "value": "yes"}, {"finding": "Have you traveled out of the country in the last 4 weeks?", "value": "N"}, {"finding": "Are you exposed to secondhand cigarette smoke on a daily basis?", "value": "yes"}, {"finding": "Do you have family members who have had lung cancer?", "value": "yes"}]}

Candidate pathologies:
- Bronchitis
- Acute pulmonary edema
- Pancreatic neoplasm
- Stable angina
- Pulmonary neoplasm
- Guillain-Barré syndrome
- Tuberculosis
- Pneumonia
- Atrial fibrillation
- Myasthenia gravis
- Anemia
- Unstable angina
- Possible NSTEMI / STEMI
- Bronchiectasis
- Myocarditis
- Pulmonary embolism
- Spontaneous rib fracture
- Acute dystonic reactions
- Bronchospasm / acute asthma exacerbation

How confident are you (0 to 1) that your chosen pathology is the correct diagnosis? Respond with ONLY valid JSON: {"Answer": "<pathology>", "Confidence": "0.XX"}
```

## Expected response format

```json
{"Answer": "Pulmonary neoplasm", "Confidence": "0.35"}
```

## Ground truth for this question

- Recorded pathology: `Pulmonary neoplasm`
- Top of differential: `Pulmonary neoplasm` (p ≈ 0.092)
- `true_probability` at scoring time = `differential[model_answer]` (0 if the
  answered pathology is absent from the differential).
