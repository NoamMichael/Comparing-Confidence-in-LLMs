"""
Generate dummy human data for the Study-2 human replication pipeline.

Real human data does not exist yet, so this fabricates a stand-in by sampling
model answers: for each question_id in a domain, it copies one row from a
randomly chosen model that answered it (one row per question). The output has
the exact column schema of the corresponding results/<DOMAIN>/<model>.csv
files, so the dummy files are drop-in "human" model files for
human_replication.py and the /llmc web tool.

Usage:
    python make_dummy_human.py                 # writes to human-data/dummy/
    python make_dummy_human.py --out some/dir  # custom output directory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from human_io import MODEL_SLUGS, RESULTS_PATH, ROOT_DIR, normalize_merge_columns

# (results subdir, output filename)
DOMAINS = [
    ("WGD", "dummy_human_wgd.csv"),
    ("WGD_SPD", "dummy_human_wgd_spd.csv"),
    ("LifeEval", "dummy_human_le.csv"),
    ("LifeEval_SPD", "dummy_human_le_spd.csv"),
]

SEED = 0


def sample_domain(domain: str, rng: np.random.Generator) -> pd.DataFrame:
    """One row per question_id, each copied from a uniformly-chosen model."""
    # Per model: a {question_id -> that model's rows} lookup, deduped implicitly
    # by grouping. Columns are normalized so every model shares one clean schema.
    per_model: dict[str, dict] = {}
    columns: list[str] | None = None
    for slug in MODEL_SLUGS:
        path = RESULTS_PATH / domain / f"{slug}.csv"
        df = normalize_merge_columns(pd.read_csv(path))
        if columns is None:
            columns = list(df.columns)
        df = df.reindex(columns=columns)
        per_model[slug] = {qid: sub for qid, sub in df.groupby("question_id", sort=False)}

    # Universe of questions = union of every model's question_ids.
    all_qids = sorted({qid for groups in per_model.values() for qid in groups})

    picked_rows = []
    for qid in all_qids:
        available = [slug for slug in MODEL_SLUGS if qid in per_model[slug]]
        slug = available[rng.integers(len(available))]
        sub = per_model[slug][qid]
        row = sub.iloc[rng.integers(len(sub))]
        picked_rows.append(row)

    out = pd.DataFrame(picked_rows, columns=columns).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT_DIR / "human-data" / "dummy",
        help="Output directory for the dummy CSVs.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    for domain, filename in DOMAINS:
        df = sample_domain(domain, rng)
        dest = args.out / filename
        df.to_csv(dest, index=False)
        print(f"{domain:14s} -> {dest}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
