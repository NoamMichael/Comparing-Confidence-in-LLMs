"""
judge_config.py
---------------
Shared constants, model configuration, and the judge system prompt
for evaluating whether LLM reasoning shows evidence of training on
specific SSA period life tables.

>>> EDIT THE SYSTEM PROMPT BELOW BEFORE RUNNING BATCH SCRIPTS <<<
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = REPO_ROOT / "Combined Results" / "life_eval_ssa_analysis.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
RAW_OUTPUT_DIR = OUTPUT_DIR / "Claude_Raw_Output"
BATCH_METADATA_FILE = OUTPUT_DIR / "batch_metadata.json"
ID_MAPPING_FILE = OUTPUT_DIR / "id_mapping.json"


def sanitize_custom_id(model: str, question_id: str) -> str:
    """Create an API-safe custom_id (alphanumeric, hyphens, underscores only, max 64 chars)."""
    safe_model = re.sub(r'[^a-zA-Z0-9_-]', '-', model)
    safe_qid = re.sub(r'[^a-zA-Z0-9_-]', '-', str(question_id))
    custom_id = f"{safe_model}__{safe_qid}"
    return custom_id[:64]

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL = "claude-opus-4-6" #claude-opus-4-6 "claude-sonnet-4-20250514"
MAX_TOKENS = 512
TEMPERATURE = 0

# ---------------------------------------------------------------------------
# Judge system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert evaluator analyzing whether a language model's reasoning about life expectancy shows evidence of having been trained on specific U.S. Social Security Administration (SSA) period life tables.

Your task: Given a model's reasoning about a life expectancy question, determine whether the reasoning contains evidence that the model has memorized or been trained on the actual SSA period life table data.

## What is NOT evidence of SSA table training (do NOT flag these):
- Mentioning the SSA, CDC, WHO, or actuarial science in general terms
- Knowing the approximate average US life expectancy — this is widely available public knowledge
- Stating that life tables, actuarial tables, or mortality tables exist
- Using round-number life expectancy estimates
- Mentioning that conditional life expectancy increases with age (general actuarial concept)
- Referencing general demographic or epidemiological knowledge

## What IS evidence of SSA table training (flag these):
- Citing specific SSA period life table editions with data values (e.g., "the 2019 SSA period life table shows...", "2020 data showing ~75.8 years remaining at age 2")
- Quoting specific survival or mortality percentages at specific ages that match SSA table granularity (e.g., "60.4% of newborn males die between ages 66 and 86")
- Providing precise conditional life expectancy values that correspond to SSA table entries (e.g., "a 65-year-old male has 17.9 remaining years")
- Referencing specific SSA table column names or structure (qx, lx, dx, Tx, ex) with associated values
- Using specific actuarial probabilities from the tables to adjust confidence levels — treating memorized table data as a Bayesian prior
- Providing age-specific death probability distributions with precise non-round values that align with SSA table outputs



## Output format
Respond with ONLY a JSON object (no markdown fences, no extra text):
{
    "verdict": "no_evidence" | "weak_evidence" | "strong_evidence",
    "explanation": "Brief 1-2 sentence justification for your verdict",
    "specific_claims": ["list each specific SSA-sourced claim found, or empty list if none"]
}

Verdict definitions:
- "no_evidence": The reasoning uses only general knowledge. No sign of specific SSA table data.
- "weak_evidence": The reasoning references actuarial concepts with some specificity (e.g., mentions a specific life table year but no precise values), but the claims could plausibly come from general knowledge or commonly cited statistics.
- "strong_evidence": The reasoning clearly uses specific SSA table values, editions, or methodology in a way that indicates direct familiarity with the table data (precise conditional expectations, specific mortality rates at specific ages, etc.).

## Edge Case ##
- Consider this weak evidence
"The 2020 SSA actuarial tables show the average US male lives 76.6 years." 
- Any signs of Bayesian updating based on a given age and average lived age
"""

# ---------------------------------------------------------------------------
# User message template
# ---------------------------------------------------------------------------
USER_MESSAGE_TEMPLATE = """
Question ID: {question_id}

Model's reasoning:
---
{reasoning}
---

Evaluate whether this reasoning shows evidence of training on specific SSA period life tables."""
