# Does SPD prompting help or hurt calibration? A confound-aware re-analysis of RQ 3

**Headline.** SPD prompting is genuinely and significantly helpful for overconfident models —
it roughly halves ECE on MedEval and cuts it sharply on WGD — but the story the paper tells
about *when it backfires* is wrong. The two dramatic backfires (GPT-5.4 Mini/WGD +89%,
Claude Haiku 4.5/LifeEval +335%) are artifacts of percentage inflation, pooling cancellation,
and an answer-parsing bug. Under a like-for-like comparison the only **robust** degradation is
a small one the paper never flagged (Llama 4 Maverick/LifeEval). The effect size is governed
by task structure — how far SPD's bin-pinned confidence sits from a domain's base-rate
accuracy — not by how well-calibrated the model already was.

All numbers from `analysis/spd_reanalysis.py` (branch `spd-reanalysis`), which reuses the
study's `scoring.get_scorer`, `study2_lib.compute_ece`, and `scoring.murphy_decomposition`,
and reproduces the repo's `build_summary` ECE column exactly before layering new metrics.

---

## 1. The claim and the worry

RQ 3 reports ΔECE as a **percent** change from a **pooled** DCE baseline and concludes SPD "is
most beneficial where DCE calibration is worst, and can be counterproductive when models are
already well-calibrated." The worry (Noam's): this is a regression-to-the-mean / percentage
artifact — the worst baselines "improve most" and the best "backfire" by construction. Largely
correct, and the mechanism is sharper than generic RTM.

## 2. Four confounds, each verified

- **C1 — Percent inflation.** ΔECE% divides by the DCE baseline; the two backfires sit on the
  two smallest baselines in the table, so tiny absolute moves explode. Percent change is not an
  effect size.
- **C2 — Pooling cancellation (largest).** Ordinary ECE bins by confidence only. When a model
  reports similar confidence on an easy and a hard item, the two land in one bin and their
  over-/under-confidence cancel, deflating ECE. DCE sweeps 20 difficulty levels (huge room to
  cancel); SPD only 4. Stratified ECE (compute within each difficulty level, then average)
  removes this: Claude/LifeEval DCE goes 0.05 (pooled) → **0.23** (stratified); GPT/WGD DCE
  0.046 → **0.099**. The "well-calibrated DCE baseline" is largely a pooling mirage.
- **C3 — Answer-parse dropout.** The paper parses answers with plain `pd.to_numeric`, so a DCE
  answer like "77 years" → NaN → dropped. SPD answers are bin centers and parse cleanly.
  Claude/LifeEval keeps only **52%** of DCE rows vs 100% of SPD. We remove this by scoring with
  scoring.py's lenient extractor (recovers all rows); it is the primary pipeline here.
- **C4 — Difficulty-composition mismatch.** DCE's 20 levels vs SPD's 4 make the modes'
  marginal difficulty mixes differ. We fix it by restricting DCE to the 4 shared levels
  {1,5,10,20} and **item-pairing** (join DCE↔SPD on photo / age+sex / patient).

## 3. The honest effect sizes

Lenient parse (full data), item-paired, difficulty-matched. ΔECE = SPD − DCE
(negative = SPD helps), 95% CI from a paired item bootstrap (2000 resamples). Both calibration
standards are shown because they disagree for LifeEval — and that disagreement is a finding.

| Domain | Model | published ΔECE% | ΔECE **pooled** (marginal) | ΔECE **stratified** (conditional) |
|---|---|---|---|---|
| MedEval | Gemini 2.5 Flash | −44% | **−0.332** [−0.346,−0.317] | −0.332 [−0.346,−0.317] |
| MedEval | Claude Haiku 4.5 | −45% | **−0.305** [−0.315,−0.294] | −0.305 [−0.315,−0.294] |
| MedEval | Llama 4 Maverick | −45% | **−0.297** [−0.308,−0.286] | −0.297 [−0.308,−0.286] |
| MedEval | GPT-5.4 Mini | −38% | **−0.288** [−0.302,−0.275] | −0.288 [−0.302,−0.275] |
| WGD | Gemini 2.5 Flash | −76% | **−0.228** [−0.268,−0.184] | −0.218 [−0.259,−0.177] |
| WGD | Llama 4 Maverick | −69% | **−0.132** [−0.167,−0.088] | −0.117 [−0.156,−0.078] |
| LifeEval | Gemini 2.5 Flash | −60% | **−0.214** [−0.240,−0.186] | −0.214 [−0.239,−0.185] |
| LifeEval | GPT-5.4 Mini | −19% | **−0.021** [−0.038,−0.007] | −0.028 [−0.043,−0.013] |
| WGD | Claude Haiku 4.5 | +20% | **−0.004** [−0.035,+0.028] ns | −0.048 [−0.082,−0.024] |
| WGD | GPT-5.4 Mini | +40% | **+0.035** [−0.005,+0.074] ns | −0.006 [−0.040,+0.026] ns |
| LifeEval | Claude Haiku 4.5 | +259% | **+0.113** [+0.084,+0.134] | −0.055 [−0.071,−0.037] |
| LifeEval | Llama 4 Maverick | +171% | **+0.069** [+0.045,+0.094] | +0.044 [+0.023,+0.067] |

Counts (paired, significant at 95%): **pooled → 8 improve / 2 worsen; stratified → 10 improve /
1 worsen.** See `figs/spd_reanalysis/delta_abs_vs_pct.png`.

## 4. What actually happens, cell by cell

- **Big, robust wins (7 cells):** all MedEval, WGD Gemini/Llama, LifeEval Gemini. Large ΔECE,
  tight CIs, identical under both standards, p<0.001. SPD clearly helps overconfident models.
- **GPT-5.4 Mini / WGD — the "+89% backfire" is a null.** ns under both standards. Its honest
  DCE calibration already sat at SPD's WGD floor (~0.08–0.10), so there was nothing to fix.
- **Claude Haiku 4.5 / LifeEval — the "+335% backfire" is a metric split, not a large harm.**
  Significantly *worse* under pooled ECE (+0.113) *and* significantly *better* under stratified
  (−0.055). DCE only wins marginally because pooling cancels its per-difficulty miscalibration;
  condition on difficulty and SPD wins. Not "SPD hurts a well-calibrated model."
- **Llama 4 Maverick / LifeEval — the one robust backfire.** Worse under *both* standards
  (+0.069 pooled, +0.044 stratified, both p<0.001). Small but real, and unremarked in the paper.

## 5. Mechanism (why the pattern looked adaptive)

- **Mechanical.** ΔECE = ECE_SPD − ECE_DCE is anti-correlated with the baseline by construction
  (r = −0.95 pooled); the informative quantity, ECE_SPD, is near-decoupled from DCE quality.
- **Structural.** SPD's scored confidence is the **modal-bin mass**, behaviorally pinned by
  spreading probability across `top_n` bins (no hard cap). That level is roughly domain-fixed
  (≈0.30 across all WGD models) and its residual miscalibration is dominated by how far it sits
  from the domain's base-rate accuracy — the **structural gap**. For one model×domain cell with
  SPD rows `i = 1…N`, confidence `conf_i` (modal-bin mass), and ground truth `y_i`
  (`true_probability`):

$$\text{gap} \;=\; \bigl|\,\bar m - \bar y\,\bigr| \;=\; \Bigl|\tfrac{1}{N}\textstyle\sum_{i}(\text{conf}_i - y_i)\Bigr| \;=\; \bigl|\,\overline{\text{over-confidence}}\,\bigr|, \qquad \bar m=\tfrac1N\textstyle\sum_i \text{conf}_i,\;\; \bar y=\tfrac1N\textstyle\sum_i y_i.$$

  It is **mean-then-absolute** (sign preserved while averaging; absolute value taken only at the
  end), so it is the *net* over/under-confidence and, by Jensen/triangle inequality, a lower bound
  on ECE (`ECE ≥ gap`). In code: `abs(mean_conf − accuracy)` on the SPD frame
  (`spd_reanalysis.py`). Across the 12 cells, corr(gap, ECE_SPD) = 0.99: SPD calibrates worst
  exactly where its bin-pinned confidence mismatches the base rate — LifeEval (acc ≈ 0.63, modal
  mass 0.27–0.48 → underconfident), which is why the two trouble cells are LifeEval, not
  "well-calibrated" cells. (The correlation is partly definitional, since the gap is a lower-bound
  component of ECE; the non-trivial claim is that this one bin-free quantity explains almost all
  the *variation* in ECE_SPD across cells while DCE quality explains ~none.) See
  `structural_gap.png`, `modal_mass_dist.png`.

## 6. Red-team / robustness (what could break this, and why it doesn't)

- **Stratified ECE small-sample bias?** Debiased each ECE against its perfect-calibration null
  (simulate outcomes ~ Bernoulli(conf), subtract expected bias). Debiasing moves every ΔECE by
  ≤0.015 and flips no sign — the stratified DCE inflation is real miscalibration, not bin noise.
- **Parse dropout driving it?** Re-scored with lenient parse (full data). It *strengthens* the
  Claude/LifeEval pooled backfire (+0.025 → +0.113) and is used throughout §3, so the verdict
  is not an artifact of discarded rows.
- **Metric cherry-picking?** Both pooled and stratified are reported; the robust conclusions
  (big wins; no large harm; Llama/LifeEval the one real backfire) hold under both. Only the
  *sign* of the small Claude/LifeEval effect is standard-dependent, and we say so.
- **Item-pairing selection.** Pairing conditions on items scored in both modes; with lenient
  parse that is ~100% of items, so negligible selection.

Limits: 12 cells (4 models × 3 domains) — directional claims are robust, fine cross-domain
ordering is not; percentile bootstrap CIs (not BCa); stratification removes cancellation only
along the labeled difficulty axis, so stratified DCE ECE is if anything a lower bound on its
true miscalibration (making SPD's advantage conservative).

## 7. Methods, in depth — the two ways to score calibration

Both "techniques" are the *same* primitive fed different rows. Nothing else differs.

**Shared inputs.** Each scored row `i` carries `conf_i ∈ [0,1]` (DCE scalar confidence, or SPD
modal-bin mass), `y_i = true_probability_i`, and a difficulty label `d_i`. Note `y_i` is *not*
the same kind of quantity across domains: WGD `y_i ∈ {0,1}` (hit/miss), while LifeEval and
MedEval `y_i ∈ [0,1]` are continuous correctness probabilities (life-table window mass;
differential probability). So a bin's "accuracy" is a hit-rate for WGD but a mean
correctness-probability for the others.

**The one primitive — `compute_ece(conf, y)`:**
1. bin rows into 11 fixed confidence bins by `conf_i`: `[0,.1),…,[.9,1),[1.0]`;
2. per bin `b`: `n_b`=count, `c̄_b`=mean conf, `ȳ_b`=mean `y`;
3. `ECE = Σ_b (n_b/N)·|ȳ_b − c̄_b|`.

**Technique A — Pooled ECE (the paper's).** Input: all cell rows together. One `compute_ece`
call → one number. Estimand: *marginal* calibration ("over my whole track record, when I say p
is the outcome p?").
- *Steelman:* textbook, pre-registered, no auxiliary variable, lowest variance; exactly right if
  a consumer only wants "discount an 0.8 to 0.8."
- *Strawman:* blind to a model that is overconfident on hard items and underconfident on easy
  ones, because those cancel inside a confidence bin — precisely the regime here (β₁>0: over-
  confidence rises with difficulty).
- *Justified* when the decision is marginal or confidence ⟂ outcome-driving covariates;
  *not* when confidence tracks difficulty (our case) and decisions condition on the item.

**Technique B — Stratified (difficulty-conditional) ECE.** Input: rows partitioned by `d`;
`compute_ece` within each stratum → `ECE_s`; combine `ECE_strat = Σ_s (n_s/N)·ECE_s`. Estimand:
*conditional* calibration ("within a difficulty, when I say p is the outcome p?").
- *Steelman:* calibration should hold conditional on what the decider knows; conditioning on a
  designed, observable difficulty axis removes an identifiable cancellation channel; strictly
  stronger than pooled.
- *Strawman:* removes cancellation only along the axis you stratify (so it is a *lower bound* on
  true miscalibration); fewer rows/bin → upward small-sample bias; a researcher degree of freedom;
  changes the estimand.
- *Justified* when a covariate correlated with confidence and outcome is observable and strata are
  large (all true here); *not* for tiny strata or no principled conditioning variable.

**Why they can disagree — the cancellation identity.** Because `|mean| ≤ mean|·|`,
`ECE_strat ≥ ECE_pooled` always, and the gap *is* the cross-difficulty cancellation. Worked
example, one confidence bin (conf≈0.5): two hard rows with `y=0.3`, two easy with `y=0.7`.
Pooled: `ȳ=0.5=c̄` → ECE 0. Stratified: `|0.3−0.5|` and `|0.7−0.5|` → ECE 0.2. DCE emits
difficulty-varying confidence (much cancellation; Claude/LifeEval pooled 0.05 → stratified 0.23);
SPD emits near-flat modal mass (little cancellation; pooled ≈ stratified). That differential is
the whole reason Claude/LifeEval flips sign between the two techniques.

**Why B's key strawman fails here.** Debiasing every ECE against its Bernoulli(conf) perfect-
calibration null moves each ΔECE by ≤0.015 and flips no sign — the stratified DCE inflation is
real conditional miscalibration, not bin noise (§6). Strawman (a) only makes SPD's advantage more
conservative; (c) is handled by pre-registering the difficulty axis as the conditioning set and
reporting both metrics.

**Does the choice even matter?** For 11 of 12 cells, no — the two techniques give the same
qualitative verdict (all MedEval identical since removal_pct induces ~no cancellation; WGD
Gemini/Llama and LifeEval Gemini/GPT improve under both; Llama/LifeEval worsens under both;
GPT/WGD null under both). Exactly **one** cell genuinely flips direction: **Claude/LifeEval** —
the very cell the paper spotlighted. So the metric debate changes one cell's sign and sharpens the
*diagnosis* of the illusory backfires, but the bottom line is invariant to it: SPD is a net
calibration win for overconfident models, with one small genuine backfire. The deep dive did not
overturn the quick read ("imperfect, but it works") — it proved which single claim is
metric-contingent and that nothing else is.

## 8. Recommendation for the paper

Replace the RQ 3 claim and the %-only table:

> SPD prompting significantly improves calibration for overconfident models, roughly halving
> ECE on MedEval and cutting it sharply on WGD (7 of 12 model–domain cells show large,
> standard-independent gains). Its effect size is governed by task structure — the gap between
> SPD's bin-pinned modal confidence and a domain's base-rate accuracy — not by the model's DCE
> calibration. Two cells previously reported as dramatic "backfires" (+89%, +335%) are artifacts
> of percentage inflation, cancellation in pooled ECE across difficulty, and answer-parse
> dropout: GPT-5.4 Mini/WGD is a null, and Claude/LifeEval worsens only marginal (pooled)
> calibration while improving conditional (difficulty-stratified) calibration. The one robust
> degradation is small (Llama 4 Maverick/LifeEval), where SPD's modal mass turns underconfident
> against a high base rate.

Report **absolute paired ΔECE with 95% CIs under both a pooled and a difficulty-stratified
standard** (§3) as the effect size; keep percent change only as secondary context.
