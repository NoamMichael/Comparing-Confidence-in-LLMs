"""
retrieve_results.py
-------------------
Retrieves a completed Anthropic Message Batch, saves raw output,
and produces an aggregated judge results CSV.

Usage:
    python retrieve_results.py demo    # retrieve demo batch
    python retrieve_results.py full    # retrieve full batch
"""

import json
import os
import sys
import re
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

from judge_config import INPUT_CSV, RAW_OUTPUT_DIR, BATCH_METADATA_FILE, OUTPUT_DIR, ID_MAPPING_FILE

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Output file mapping
OUTPUT_MAP = {
    "demo": {
        "raw": RAW_OUTPUT_DIR / "demo_batch_raw.jsonl",
        "csv": OUTPUT_DIR / "demo_judge_results.csv",
    },
    "full": {
        "raw": RAW_OUTPUT_DIR / "judge_batch_raw.jsonl",
        "csv": OUTPUT_DIR / "judge_results.csv",
    },
}


def parse_verdict(text: str) -> dict:
    """Extract JSON verdict from the assistant's response text."""
    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown fences or surrounding text
    match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text as explanation
    return {
        "verdict": "parse_error",
        "explanation": text[:200],
        "specific_claims": []
    }


def resolve_custom_id(custom_id: str, id_mapping: dict) -> tuple:
    """Resolve sanitized custom_id back to (Model, Question ID) using the mapping."""
    if custom_id in id_mapping:
        entry = id_mapping[custom_id]
        return entry["Model"], entry["Question ID"]
    # Fallback: split on double underscore
    parts = custom_id.split("__", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return custom_id, ""


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("demo", "full"):
        print("Usage: python retrieve_results.py [demo|full]")
        sys.exit(1)

    batch_type = sys.argv[1]
    outputs = OUTPUT_MAP[batch_type]

    # Load ID mapping
    id_mapping = {}
    if ID_MAPPING_FILE.exists():
        with open(ID_MAPPING_FILE, "r") as f:
            id_mapping = json.load(f)

    # Load batch ID
    if not BATCH_METADATA_FILE.exists():
        print(f"Error: {BATCH_METADATA_FILE} not found. Run submit script first.")
        sys.exit(1)

    with open(BATCH_METADATA_FILE, "r") as f:
        metadata = json.load(f)

    if batch_type not in metadata:
        print(f"Error: No '{batch_type}' batch found in metadata. Run the submit script first.")
        sys.exit(1)

    batch_id = metadata[batch_type]["batch_id"]
    print(f"Retrieving {batch_type} batch: {batch_id}")

    # Check status
    client = Anthropic()
    batch_info = client.messages.batches.retrieve(batch_id)
    print(f"  Status: {batch_info.processing_status}")

    if batch_info.processing_status != "ended":
        print(f"\n  Batch is not yet complete.")
        if hasattr(batch_info, 'request_counts'):
            counts = batch_info.request_counts
            print(f"  Processing: {counts.processing}, Succeeded: {counts.succeeded}, "
                  f"Errored: {counts.errored}, Canceled: {counts.canceled}, Expired: {counts.expired}")
        print("  Re-run this script later.")
        sys.exit(0)

    # Stream results and save raw JSONL
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = outputs["raw"]

    print(f"  Streaming results to {raw_path}...")
    results = []
    with open(raw_path, "w", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch_id):
            raw_line = result.model_dump_json()
            f.write(raw_line + "\n")
            results.append(result)

    print(f"  Saved {len(results):,} raw results")

    # Parse verdicts
    print("  Parsing verdicts...")
    rows = []
    parse_errors = 0
    for result in results:
        model, question_id = resolve_custom_id(result.custom_id, id_mapping)

        if result.result.type == "succeeded":
            # Extract text from the response
            text_blocks = [
                block.text for block in result.result.message.content
                if block.type == "text"
            ]
            response_text = " ".join(text_blocks)
            verdict_data = parse_verdict(response_text)
            if verdict_data.get("verdict") == "parse_error":
                parse_errors += 1
        else:
            verdict_data = {
                "verdict": "api_error",
                "explanation": f"Result type: {result.result.type}",
                "specific_claims": []
            }

        rows.append({
            "Model": model,
            "Question ID": question_id,
            "verdict": verdict_data.get("verdict", "unknown"),
            "explanation": verdict_data.get("explanation", ""),
            "specific_claims": json.dumps(verdict_data.get("specific_claims", [])),
        })

    # Build dataframe
    judge_df = pd.DataFrame(rows)

    # Merge with original data to get matched_keywords and references_ssa_actuarial
    original_df = pd.read_csv(INPUT_CSV, low_memory=False)
    original_df["Question ID"] = original_df["Question ID"].astype(str)
    judge_df["Question ID"] = judge_df["Question ID"].astype(str)

    merge_cols = ["Model", "Question ID", "Reasoning", "matched_keywords", "references_ssa_actuarial"]
    available_cols = [c for c in merge_cols if c in original_df.columns]
    merged = judge_df.merge(
        original_df[available_cols],
        on=["Model", "Question ID"],
        how="left"
    )

    # Reorder columns
    col_order = ["Model", "Question ID", "Reasoning", "verdict", "explanation",
                 "specific_claims", "matched_keywords", "references_ssa_actuarial"]
    col_order = [c for c in col_order if c in merged.columns]
    merged = merged[col_order]

    # Save CSV
    csv_path = outputs["csv"]
    merged.to_csv(csv_path, index=False)
    print(f"  Saved aggregated results to {csv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY ({batch_type})")
    print(f"{'='*60}")
    print(f"  Total results: {len(merged):,}")
    if parse_errors > 0:
        print(f"  Parse errors:  {parse_errors}")

    verdict_counts = merged["verdict"].value_counts()
    print(f"\n  Verdict distribution:")
    for verdict, count in verdict_counts.items():
        pct = 100 * count / len(merged)
        print(f"    {verdict:<20} {count:>6,} ({pct:5.1f}%)")

    print(f"\n  Per-model breakdown:")
    model_verdicts = merged.groupby("Model")["verdict"].value_counts().unstack(fill_value=0)
    print(model_verdicts.to_string())
    print()


if __name__ == "__main__":
    main()
