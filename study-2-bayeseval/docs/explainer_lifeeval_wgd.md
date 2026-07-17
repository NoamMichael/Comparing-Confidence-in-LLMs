# Explainer: LifeEval & WGD — DCE and SPD

How the two continuous-answer domains in Study 2 (BayesEval) were constructed,
administered, and scored, in each of the two elicitation modes. For the verbatim
prompt templates and runner wrappers see [prompts.md](prompts.md); for the raw
build logic see each domain's `build_benchmark.py` / `build_spd_benchmark.py`.

## The common design

Every question set follows the same recipe:

1. **A point-estimation question** the model must answer with a number (age at
   death, or weight in pounds).
2. **A confidence elicitation** attached to that question. How it is elicited is
   the experimental manipulation:
   - **DCE (Direct Confidence Elicitation):** the model gives one point estimate
     and one scalar probability that its estimate falls within a stated window
     of the truth. Difficulty is swept by varying the window size across
     questions (20 levels).
   - **SPD (Subjective Probability Distribution):** the model reports its top-N
     most likely ranges ("bins") of a stated width, with a probability on each.
     The model chooses where to place the bins; only the width is fixed (4
     levels). One response therefore encodes a whole distribution.
3. **Ground truth is a probability, not (only) a right answer.** Calibration is
   measured by comparing the stated confidence to the actual probability that
   the answer landed in the window.

Administration was identical across sets: each question was sent **once per
model** as a **single `user` message with no system prompt** to four models via
OpenRouter (Claude Haiku 4.5, Gemini 2.5 Flash, Llama 4 Maverick, GPT-5.4 Mini).
DCE questions have two runs — the original (`results/`) and a 2026-07 re-run
whose JSON wrapper also requests step-by-step reasoning (`results_reasoning/`).
SPD prompts are identical in both runs.

---

## LifeEval — conditional life expectancy

**The question (both modes):**

> Given that an American male has lived at least {age} years, estimate how old
> he will be when he dies.

Items are the full crossing of **age attained ∈ {0, …, 100} × sex ∈ {male,
female}** — 202 base items. There is no sampling; every cell of the grid is
asked.

**Ground truth** comes directly from the 2022 US Period Life Table's per-year
death probabilities (ages 0–118, separately per sex). The probability that a
person who reached age *a* dies inside the integer-age window
`[floor(y−r), ceil(y+r))` is summed straight from the table, so a model's
confidence can be scored against the *true conditional probability* of its own
answer — no single "right" age at death is assumed.

### LifeEval DCE — 4,040 questions

Each base item is crossed with an integer **radius r ∈ {1, …, 20} years**
(101 ages × 2 sexes × 20 radii = 4,040). The confidence prompt is:

> How certain are you that your answer is within {radius} years of the true value?

The model returns `{"Answer": …, "Confidence": …}`. Scoring computes
`true_probability` = P(death in [Answer − r, Answer + r] | survived to age) from
the life table, and Brier score = (Confidence − true_probability)².

Difficulty varies along two axes:

- **Radius (the manipulated axis):** a ±1-year window is objectively hard even
  for a perfectly calibrated forecaster; ±20 years is easy.
- **Age (intrinsic):** conditional lifespan distributions are wide at birth and
  narrow at old age, so the same radius captures more probability mass for a
  95-year-old than for a newborn.

Because the achievable probability differs per cell, the benchmark precomputes
for every (age, sex, radius) the **best possible answer** (the window placement
maximizing captured probability) and the **MAS (maximum achievable score)** —
the probability that optimal answer captures. MAS is the ceiling a perfectly
calibrated, perfectly knowledgeable responder could reach: mean MAS is ≈0.12 at
radius 1 and ≈0.94 at radius 20. A model stating 0.9 confidence on a radius-1
question is therefore overconfident no matter what age it answered.

### LifeEval SPD — 808 questions

Same 202 base items, crossed with **bin width ∈ {2, 10, 20, 40} years** (top-N
requested: 10, 10, 5, and 3 bins respectively). The confidence prompt (e.g.
width 20):

> Using age ranges of exactly 20 years (e.g. [70, 90)), report your top 5 most
> likely ranges where this person will die. Each range must be exactly 20 years
> wide. You choose where to place them. … Respond with ONLY a JSON array …
> `[{"min": …, "max": …, "confidence": …}, …]`

Bins are **model-placed** (no fixed grid) to avoid boundary artifacts; only the
width is constrained. For DCE-comparable scoring, the parser extracts the
**modal bin** (highest stated confidence), takes its midpoint as the Answer and
its confidence as the Confidence, and scores it exactly like a DCE question with
`radius = bin_width / 2`. The four widths therefore correspond to DCE radii
{1, 5, 10, 20} — matched cells exist in both modes at those radii.

Why only 4 widths instead of 20? One SPD response already contains an entire
distribution (up to 10 bins with probabilities), so the 20-point sweep DCE needs
to trace out the confidence–window relationship would be largely redundant; the
4 widths span the DCE radius range (both endpoints plus two interior points) at
a fifth of the query cost.

---

## WGD — weight guessing from photos

**The task:** the model sees a photo of a person (attached to the message as a
base64 image) and must estimate their weight in pounds. Items are **231 photos**
of consenting participants with measured ground-truth weights (mean ≈150 lbs,
range 18–380). One photo (`269.jpg`) accidentally entered only the SPD build
because its label/photo pairing became available between the two builds; it is
**excluded from all analysis** so both modes run on the identical 231-photo set.

Unlike LifeEval there is no population model: ground truth is **deterministic**.
The model's guess either landed within the tolerance of the measured weight
(outcome = 1) or it didn't (outcome = 0). Calibration compares stated confidence
to the empirical hit rate.

### WGD DCE — 4,620 questions

Each photo is crossed with an integer **tolerance ∈ {1, …, 20} lbs**
(231 × 20 = 4,620). The prompt (which also embeds the elicitation):

> Look at this photo of a person. You MUST estimate their weight in pounds. Do
> not refuse or abstain — give your best guess even if uncertain. How confident
> are you (0 to 1) that your estimate is within {within_lbs} lbs of their true
> weight? If you are very unsure, use a low confidence score, but you must still
> provide a weight estimate.

(The "must not refuse" framing exists because vision models otherwise decline to
estimate people's weight.) Difficulty again has two axes: the tolerance (±1 lb
is near-impossible from a photo; ±20 lbs is achievable) and the photo itself
(some bodies/clothing/poses are harder to judge).

### WGD SPD — 924 questions analyzed

Each photo crossed with **bin width ∈ {2, 10, 20, 40} lbs**, same top-N scheme
as LifeEval (231 × 4 = 924 after the `269.jpg` exclusion; the committed
benchmark CSV contains 928). The model reports its top-N most likely weight
ranges of the given width, model-placed. Scoring again takes the modal bin's
midpoint and confidence and scores deterministically with
`within_lbs = bin_width / 2` — i.e. matched to DCE tolerances {1, 5, 10, 20}.

---

## Side-by-side summary

| | LifeEval DCE | LifeEval SPD | WGD DCE | WGD SPD |
|---|---|---|---|---|
| Base items | 101 ages × 2 sexes | same | 231 photos | same (after exclusion) |
| Difficulty knob | radius 1–20 yr | bin width {2,10,20,40} yr | tolerance 1–20 lbs | bin width {2,10,20,40} lbs |
| Top-N bins allowed | — | tied to width: {10,10,5,3} | — | tied to width: {10,10,5,3} |
| Prompts | 202 × 20 radii = 4,040 | 202 × 4 widths = 808 | 231 × 20 tolerances = 4,620 | 231 × 4 widths = 924 |
| Response | point + scalar confidence | top-N bins + probabilities | point + scalar confidence | top-N bins + probabilities |
| Ground truth | empirical life-table conditional probability | same, on modal bin | binary hit within tolerance | same, on modal bin |
| DCE↔SPD matched cells | — | radii {1, 5, 10, 20} | — | tolerances {1, 5, 10, 20} |

Caveats to keep in mind when analyzing:

- **Paired DCE-vs-SPD comparisons** are only possible at the four matched
  radii/tolerances; the other 16 DCE levels have no SPD counterpart.
- **SPD scoring uses only the modal bin.** The rest of the reported distribution
  is currently unused by the headline calibration metrics.
- **LifeEval confidences are scored against population statistics**
  (the SSA life table), not an observed death — a perfectly calibrated responder's
  Brier score is bounded by the MAS, which varies by cell.
- **WGD outcomes are noisy at the item level** (binary), so per-cell accuracy
  estimates average over photos, while LifeEval's continuous true probabilities
  do not need that averaging.
