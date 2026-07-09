#!/usr/bin/env python3
"""
Build the MedEval benchmark CSV from the DDXPlus English release.

Each DDXPlus patient carries a differential diagnosis — a distribution over
pathologies that is the analytical ground truth for this domain. For each
sampled patient we emit a question that presents the patient's age, sex, and
symptom list (translated from E-codes via release_evidences.json) alongside
the candidate pathologies (shuffled). The model picks a single diagnosis and
reports a confidence.

Difficulty is controlled by removing candidate pathologies and renormalizing
the differential. Fewer candidates = easier task.

Columns written:
    question_id, question_prompt, confidence_prompt, age, sex,
    true_pathology, differential_json, removal_pct

true_probability is computed at scoring time as differential[model_answer].

Usage:
    python build_benchmark.py --ddxplus ../../../ddxplus/22687585 --n 500 \
        --min-candidates 10 --removal-pcts 0,10,25,50
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
    "below, pick the single most likely pathology from the candidate list. "
    "You MUST commit to one diagnosis — do not hedge or list alternatives.\n\n"
    "{vignette}\n\nCandidate pathologies:\n{candidates}"
)

CONFIDENCE_INSTRUCTION = (
    "There are {n_candidates} candidate pathologies. Estimate the probability "
    "(0 to 1) that your chosen pathology is the correct diagnosis for this "
    "patient, given only the symptoms and candidates provided. A uniform prior "
    "would assign {uniform_prior:.2f} to each candidate. "
    "Respond with ONLY valid JSON: "
    '{{"Answer": "<pathology>", "Confidence": "0.XX"}}'
)


def load_evidence_map(evidences_path: Path) -> dict:
    with open(evidences_path) as f:
        return json.load(f)


def render_evidence(code: str, emap: dict) -> dict:
    """Translate an EVIDENCES entry (possibly E_x_@_V_y or E_x_@_n) to a dict."""
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
    """Remove pct_remove% of candidate pathologies and renormalize.

    Always keeps the true pathology. Removes the least-likely candidates
    (after excluding the true pathology from removal consideration).
    Returns the reduced, renormalized differential.
    """
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


def write_variant(pool, pct, emap, split, seed, outpath):
    """Write a benchmark CSV with pct% of candidate pathologies removed."""
    rng_variant = random.Random(seed + pct)
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id", "question_prompt", "confidence_prompt",
            "age", "sex", "true_pathology", "differential_json",
        ])
        for i, p in enumerate(pool):
            evidences = ast.literal_eval(p["EVIDENCES"])
            differential = ast.literal_eval(p["DIFFERENTIAL_DIAGNOSIS"])

            reduced_diff = reduce_candidates(
                differential, p["PATHOLOGY"], pct, rng_variant)

            vignette = build_vignette(p["AGE"], p["SEX"], evidences, emap)
            candidates = [name for name, _ in reduced_diff]
            rng_variant.shuffle(candidates)
            candidates_str = "\n".join(f"- {c}" for c in candidates)
            n_candidates = len(candidates)
            w.writerow([
                f"med_{split}_{i:05d}_c{pct}",
                PATHOLOGY_INSTRUCTION.format(
                    vignette=vignette, candidates=candidates_str),
                CONFIDENCE_INSTRUCTION.format(
                    n_candidates=n_candidates,
                    uniform_prior=1.0 / n_candidates),
                p["AGE"], p["SEX"], p["PATHOLOGY"],
                json.dumps(reduced_diff),
            ])
    return len(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddxplus", type=Path,
                    default=Path(__file__).parent / "Data",
                    help="Directory containing release_evidences.json and release_<split>_patients.zip")
    ap.add_argument("--split", default="test",
                    choices=["train", "validate", "test"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).parent / "Data")
    ap.add_argument("--min-findings", type=int, default=0,
                    help="Keep only patients with >= this many findings")
    ap.add_argument("--min-candidates", type=int, default=0,
                    help="Keep only patients with >= this many candidates in differential")
    ap.add_argument("--removal-pcts", default="0",
                    help="Comma-separated candidate removal percentages (e.g. 0,10,25,50)")
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

    pcts = [int(x) for x in args.removal_pcts.split(",")]
    args.outdir.mkdir(parents=True, exist_ok=True)

    variant_paths = []
    for pct in pcts:
        outpath = args.outdir / f"benchmark_remove{pct}.csv"
        n = write_variant(pool, pct, emap, args.split, args.seed, outpath)
        variant_paths.append(outpath)
        print(f"Wrote {n} questions (candidates -{pct}%) to {outpath}")

    if len(pcts) > 1:
        combined = args.outdir / "benchmark_combined.csv"
        with open(combined, "w", newline="") as fout:
            w = csv.writer(fout)
            w.writerow([
                "question_id", "question_prompt", "confidence_prompt",
                "age", "sex", "true_pathology", "differential_json",
                "removal_pct",
            ])
            for pct, vpath in zip(pcts, variant_paths):
                with open(vpath) as fin:
                    reader = csv.reader(fin)
                    next(reader)
                    for row in reader:
                        w.writerow(row + [str(pct)])
        print(f"Wrote combined benchmark ({len(pool) * len(pcts)} questions) to {combined}")


if __name__ == "__main__":
    main()
