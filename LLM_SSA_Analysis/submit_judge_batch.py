"""
submit_judge_batch.py
---------------------
Creates and submits a full Anthropic Message Batch for all rows
in life_eval_ssa_analysis.csv (~8,261 rows).

Usage:
    python submit_judge_batch.py
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

    # Build requests
    print(f"Building batch requests for all {len(df):,} rows...")
    requests, id_mapping = build_requests(df)
    print(f"  Built {len(requests):,} requests")

    # Submit batch
    client = Anthropic()
    try:
        batch_job = client.messages.batches.create(requests=requests)
        print(f"Batch submitted successfully!")
        print(f"  Batch ID: {batch_job.id}")
        print(f"  Status:   {batch_job.processing_status}")

        # Save metadata
        metadata = {}
        if BATCH_METADATA_FILE.exists():
            with open(BATCH_METADATA_FILE, "r") as f:
                metadata = json.load(f)

        metadata["full"] = {
            "batch_id": batch_job.id,
            "row_count": len(df),
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
