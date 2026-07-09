#!/usr/bin/env python3
"""
Build the WGD SPD benchmark CSV for RQ3.

Instead of point estimate + scalar confidence, the SPD prompt asks models to
report their top-N most likely weight bins of a specified width.
Model-driven bin placement: the model chooses where to place bins (no fixed
grid). The only constraint is that each bin must be exactly `bin_width` lbs
wide.

For each (photo, bin_width in {2, 10, 20, 40}):
  - question_prompt: asks model to estimate weight (no tolerance mentioned)
  - confidence_prompt: specifies bin width, requests top-N JSON array
  - within_lbs = bin_width / 2 (so existing score_wgd works on the modal bin)

Output columns:
    question_id, question_prompt, confidence_prompt, photo,
    within_lbs, true_weight, bin_width, top_n

Usage:
    python build_spd_benchmark.py
"""

import argparse
import csv
from pathlib import Path


BIN_WIDTHS = [2, 10, 20, 40]

TOP_N = {2: 10, 10: 10, 20: 5, 40: 3}

QUESTION_PROMPT = (
    "Look at this photo of a person. You MUST estimate their weight in pounds. "
    "Do not refuse or abstain — give your best guess even if uncertain."
)

CONFIDENCE_TEMPLATE = (
    "Using weight ranges of exactly {bw} lbs (e.g. [150, {bw_example})), "
    "report your top {top_n} most likely ranges for this person's weight. "
    "Each range must be exactly {bw} lbs wide. You choose where to place them.\n\n"
    "Respond with ONLY a JSON array in this exact format:\n"
    '[{{"min": <low_weight>, "max": <high_weight>, "confidence": <probability between 0 and 1>}}, ...]\n'
    "No other text."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    default=Path(__file__).parent / "Data" / "labels.csv")
    ap.add_argument("--photos", type=Path,
                    default=Path(__file__).parent / "Data" / "Photos")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "Data" / "benchmark_spd.csv")
    args = ap.parse_args()

    available = {p.stem for p in args.photos.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    with open(args.labels) as f:
        labels = [r for r in csv.DictReader(f)
                  if r.get("Weight") and r["Participant Number"] in available]

    qid = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id", "question_prompt", "confidence_prompt",
            "photo", "within_lbs", "true_weight", "bin_width", "top_n",
        ])
        for row in labels:
            pid = row["Participant Number"]
            weight = float(row["Weight"])
            for bw in BIN_WIDTHS:
                top_n = TOP_N[bw]
                c_prompt = CONFIDENCE_TEMPLATE.format(
                    bw=bw, bw_example=150 + bw, top_n=top_n,
                )
                w.writerow([
                    f"wgd_spd_{pid}_{bw:02d}",
                    QUESTION_PROMPT, c_prompt,
                    f"{pid}.jpg", bw // 2, weight, bw, top_n,
                ])
                qid += 1

    print(f"Wrote {qid} questions to {args.out}")


if __name__ == "__main__":
    main()
