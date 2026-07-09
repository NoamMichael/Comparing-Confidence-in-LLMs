#!/usr/bin/env python3
"""
Build the WGD (Weight Guessing Dataset) benchmark CSV.

For each labeled photo and each tolerance in `--within-lbs`, emit a question
asking the model to estimate the subject's weight in pounds along with a
confidence that the estimate falls within `within_lbs` of the true weight.

Ground truth is deterministic: `true_probability = 1` if the model's weight
guess is within `within_lbs` of the measured weight, else `0`. Brier Score on
the stated confidence therefore measures how well the model's internal sense
of "am I close?" tracks whether it actually was.

Output columns:
    question_id, question_prompt, confidence_prompt, photo, within_lbs, true_weight
"""

import argparse
import csv
from pathlib import Path


PROMPT_TEMPLATE = (
    "Look at this photo of a person. You MUST estimate their weight in pounds. "
    "Do not refuse or abstain — give your best guess even if uncertain. "
    "How confident are you (0 to 1) that your estimate is within {within_lbs} lbs "
    "of their true weight? If you are very unsure, use a low confidence score, "
    "but you must still provide a weight estimate."
)

CONFIDENCE_INSTRUCTION = (
    "Respond with ONLY valid JSON in this exact format: "
    '{"Answer": "<weight in lbs>", "Confidence": "0.XX"}'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    default=Path(__file__).parent / "Data" / "labels.csv")
    ap.add_argument("--photos", type=Path,
                    default=Path(__file__).parent / "Data" / "Photos")
    ap.add_argument("--within-lbs", type=int, nargs="+",
                    default=list(range(1, 21)))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "Data" / "benchmark.csv")
    args = ap.parse_args()

    available = {p.stem for p in args.photos.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    with open(args.labels) as f:
        labels = [r for r in csv.DictReader(f)
                  if r.get("Weight") and r["Participant Number"] in available]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id", "question_prompt", "confidence_prompt",
            "photo", "within_lbs", "true_weight",
        ])
        for row in labels:
            pid = row["Participant Number"]
            weight = float(row["Weight"])
            for wl in args.within_lbs:
                w.writerow([
                    f"wgd_{pid}_{wl:02d}",
                    PROMPT_TEMPLATE.format(within_lbs=wl),
                    CONFIDENCE_INSTRUCTION,
                    f"{pid}.jpg", wl, weight,
                ])
    print(f"Wrote {len(labels) * len(args.within_lbs)} questions to {args.out}")


if __name__ == "__main__":
    main()
