#!/usr/bin/env python3
"""
Build the MedEval SPD benchmark CSV for RQ3.

Instead of picking one diagnosis + scalar confidence, the SPD prompt asks
models to assign a probability to each candidate pathology. The candidate
list IS the natural "grid" — no binning needed. The modal candidate's
probability becomes "Confidence" for Brier scoring.

Reuses the same patient pool, vignette rendering, and candidate-removal logic
as build_benchmark.py. Supports evidence removal variants (--removal-pcts)
and candidate removal variants (--candidate-removal-pcts).

Columns written:
    question_id, question_prompt, confidence_prompt, age, sex,
    true_pathology, differential_json, removal_pct

Usage:
    python build_spd_benchmark.py --ddxplus Data --n 500
    python build_spd_benchmark.py --ddxplus Data --n 500 --candidate-removal-pcts 0,10,25,50
"""

import argparse
import ast
import csv
import json
import math
import random
import zipfile
from pathlib import Path


PATHOLOGY_INSTRUCTION = (
    "You are a diagnostic reasoning assistant. Based on the patient vignette "
    "below, estimate the probability that each candidate pathology is the "
    "correct diagnosis. You MUST assign a probability to every candidate.\n\n"
    "{vignette}\n\nCandidate pathologies:\n{candidates}"
)

CONFIDENCE_INSTRUCTION = (
    "For each candidate pathology, report your estimated probability that it "
    "is the correct diagnosis. Probabilities should sum to 1.0.\n\n"
    "Respond with ONLY a JSON array in this exact format:\n"
    '[{{"pathology": "<name>", "confidence": <probability between 0 and 1>}}, ...]\n'
    "No other text."
)


def load_evidence_map(evidences_path: Path) -> dict:
    with open(evidences_path) as f:
        return json.load(f)


def render_evidence(code: str, emap: dict) -> dict:
    if "_@_" in code:
        ecode, value = code.split("_@_", 1)
    else:
        ecode, value = code, None
    entry = emap.get(ecode)
    if entry is None:
        return {"finding": code, "value": "yes"}
    question = entry.get("question_en", ecode)
    if value is None:
        return {"finding": question, "value": "yes"}
    vm = entry.get("value_meaning") or {}
    meaning = vm.get(value, {}).get("en", value) if isinstance(vm, dict) else value
    return {"finding": question, "value": meaning}


def build_vignette(age, sex, evidences, emap) -> str:
    sex_word = {"M": "male", "F": "female"}.get(sex, sex)
    vignette = {
        "patient": {"age": int(age), "sex": sex_word},
        "findings": [render_evidence(c, emap) for c in evidences],
    }
    return json.dumps(vignette)


def iter_patients(patients_zip: Path):
    with zipfile.ZipFile(patients_zip) as zf:
        inner = next(n for n in zf.namelist() if not n.endswith("/"))
        with zf.open(inner) as f:
            text = f.read().decode("utf-8").splitlines()
    reader = csv.DictReader(text)
    for row in reader:
        yield row


def reduce_candidates(differential, true_pathology, pct_remove, rng):
    if pct_remove <= 0 or len(differential) <= 1:
        return differential
    n_remove = math.ceil(len(differential) * pct_remove / 100)
    max_remove = len(differential) - 1
    n_remove = min(n_remove, max_remove)
    removable = [(name, prob) for name, prob in differential if name != true_pathology]
    removable.sort(key=lambda x: x[1])
    to_remove = set(name for name, _ in removable[:n_remove])
    reduced = [(name, prob) for name, prob in differential if name not in to_remove]
    total = sum(prob for _, prob in reduced)
    if total > 0:
        reduced = [(name, prob / total) for name, prob in reduced]
    return reduced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddxplus", type=Path,
                    default=Path(__file__).parent / "Data")
    ap.add_argument("--split", default="test",
                    choices=["train", "validate", "test"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "Data" / "benchmark_spd.csv")
    ap.add_argument("--min-findings", type=int, default=0)
    ap.add_argument("--min-candidates", type=int, default=0)
    ap.add_argument("--candidate-removal-pcts", default="0,10,25,50",
                    help="Comma-separated candidate removal percentages")
    args = ap.parse_args()

    emap = load_evidence_map(args.ddxplus / "release_evidences.json")
    patients_zip = args.ddxplus / f"release_{args.split}_patients.zip"

    rng = random.Random(args.seed)
    pool = list(iter_patients(patients_zip))
    rng.shuffle(pool)
    pool = pool[: args.n]

    if args.min_findings > 0:
        pool = [p for p in pool
                if len(ast.literal_eval(p["EVIDENCES"])) >= args.min_findings]
        print(f"Filtered to {len(pool)} patients with >= {args.min_findings} findings")

    if args.min_candidates > 0:
        pool = [p for p in pool
                if len(ast.literal_eval(p["DIFFERENTIAL_DIAGNOSIS"])) >= args.min_candidates]
        print(f"Filtered to {len(pool)} patients with >= {args.min_candidates} candidates")

    cpcts = [int(x) for x in args.candidate_removal_pcts.split(",")]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    qid = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id", "question_prompt", "confidence_prompt",
            "age", "sex", "true_pathology", "differential_json",
            "removal_pct",
        ])

        for pct in cpcts:
            rng_variant = random.Random(args.seed + pct)
            for i, p in enumerate(pool):
                evidences = ast.literal_eval(p["EVIDENCES"])
                differential = ast.literal_eval(p["DIFFERENTIAL_DIAGNOSIS"])

                reduced_diff = reduce_candidates(
                    differential, p["PATHOLOGY"], pct, rng_variant)

                vignette = build_vignette(p["AGE"], p["SEX"], evidences, emap)
                candidates = [name for name, _ in reduced_diff]
                rng_variant.shuffle(candidates)
                candidates_str = "\n".join(f"- {c}" for c in candidates)

                w.writerow([
                    f"med_spd_{args.split}_{i:05d}_c{pct}",
                    PATHOLOGY_INSTRUCTION.format(
                        vignette=vignette, candidates=candidates_str),
                    CONFIDENCE_INSTRUCTION,
                    p["AGE"], p["SEX"], p["PATHOLOGY"],
                    json.dumps(reduced_diff),
                    pct,
                ])
                qid += 1

    print(f"Wrote {qid} questions to {args.out}")


if __name__ == "__main__":
    main()
