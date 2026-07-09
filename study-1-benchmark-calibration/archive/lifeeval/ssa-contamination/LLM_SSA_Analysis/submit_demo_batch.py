"""
submit_demo_batch.py
--------------------
Creates and submits a small demo Anthropic Message Batch (~100 rows)
for validating the judge prompt before running the full batch.

Stratified sampling: ~9 rows per model, with a mix of keyword-flagged
and unflagged rows when available.

Usage:
    python submit_demo_batch.py
"""

import json
import os
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic, APIError
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from judge_config import (
    INPUT_CSV, MODEL, MAX_TOKENS, TEMPERATURE,
    SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE, BATCH_METADATA_FILE,
    ID_MAPPING_FILE, sanitize_custom_id
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

DEMO_SIZE = 100


def sample_stratified(df: pd.DataFrame, total_n: int = DEMO_SIZE) -> pd.DataFrame:
    """Sample ~total_n rows stratified by model, mixing flagged and unflagged."""
    models = df["Model"].unique()
    per_model = max(1, total_n // len(models))
    samples = []

    for model in models:
        model_df = df[df["Model"] == model]
        flagged = model_df[model_df["references_ssa_actuarial"] == True]
        unflagged = model_df[model_df["references_ssa_actuarial"] == False]

        # Try to get a mix: 2/3 flagged, 1/3 unflagged
        n_flagged = min(len(flagged), (per_model * 2) // 3)
        n_unflagged = min(len(unflagged), per_model - n_flagged)
        # Fill remaining from whichever has more
        n_flagged = min(len(flagged), per_model - n_unflagged)

        if n_flagged > 0:
            samples.append(flagged.sample(n=n_flagged, random_state=42))
        if n_unflagged > 0:
            samples.append(unflagged.sample(n=n_unflagged, random_state=42))

    result = pd.concat(samples, ignore_index=True)
    print(f"  Sampled {len(result)} rows across {len(models)} models")
    return result


def build_requests(df: pd.DataFrame) -> tuple:
    """Build Anthropic batch Request objects and an ID mapping from the dataframe."""
    requests = []
    id_mapping = {}  # sanitized_id -> {Model, Question ID}
    for _, row in df.iterrows():
        model_name = str(row.get("Model", "unknown"))
        question_id = str(row.get("Question ID", ""))
        reasoning = str(row.get("Reasoning", ""))

        custom_id = sanitize_custom_id(model_name, question_id)
        id_mapping[custom_id] = {"Model": model_name, "Question ID": question_id}

        user_message = USER_MESSAGE_TEMPLATE.format(
            model_name=model_name,
            question_id=question_id,
            reasoning=reasoning
        )

        requests.append(
            Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": user_message}],
                    system=SYSTEM_PROMPT,
                    temperature=TEMPERATURE
                )
            )
        )
    return requests, id_mapping


def main():
    # Load data
    print(f"Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  Total rows: {len(df):,}")

    # Stratified sample
    print(f"Sampling ~{DEMO_SIZE} rows (stratified by model)...")
    demo_df = sample_stratified(df, DEMO_SIZE)

    # Show sample breakdown
    breakdown = demo_df.groupby("Model")["references_ssa_actuarial"].agg(
        total="count", flagged="sum"
    )
    print("\n  Per-model sample:")
    for model, row in breakdown.iterrows():
        print(f"    {model:<35} {int(row['total']):>3} rows ({int(row['flagged'])} flagged)")

    # Build requests
    print(f"\nBuilding {len(demo_df)} batch requests...")
    requests, id_mapping = build_requests(demo_df)

    # Submit batch
    client = Anthropic()
    try:
        batch_job = client.messages.batches.create(requests=requests)
        print(f"\nDemo batch submitted successfully!")
        print(f"  Batch ID: {batch_job.id}")
        print(f"  Status:   {batch_job.processing_status}")

        # Save metadata
        metadata = {}
        if BATCH_METADATA_FILE.exists():
            with open(BATCH_METADATA_FILE, "r") as f:
                metadata = json.load(f)

        metadata["demo"] = {
            "batch_id": batch_job.id,
            "row_count": len(demo_df),
            "model": MODEL,
        }

        with open(BATCH_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved batch ID to {BATCH_METADATA_FILE}")

        # Save ID mapping for retrieval
        existing_mapping = {}
        if ID_MAPPING_FILE.exists():
            with open(ID_MAPPING_FILE, "r") as f:
                existing_mapping = json.load(f)
        existing_mapping.update(id_mapping)
        with open(ID_MAPPING_FILE, "w") as f:
            json.dump(existing_mapping, f, indent=2)
        print(f"  Saved ID mapping to {ID_MAPPING_FILE}")

    except APIError as e:
        print(f"Error submitting batch: {e}")
        raise


if __name__ == "__main__":
    main()
