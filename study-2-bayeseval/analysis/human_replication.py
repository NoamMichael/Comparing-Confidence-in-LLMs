"""
Plug-and-play Study-2 replication with humans as a fifth "model".

Point this at a human replication of the WGD and/or LifeEval tasks (direct and/or
SPD variants) and it regenerates the full Study-2 figure set and summary table,
with a "Humans" series added wherever human data is supplied. The four LLM
results are bundled in results/; the collaborator only provides human CSV(s).

Each human CSV needs at least `question_id, Answer, Confidence`; any missing
ground-truth columns are back-filled from the domain benchmark. Domains without a
human file (always MedEval, plus any variant you omit) render models-only.

Example:
    python human_replication.py \\
        --wgd    ../human-data/dummy/dummy_human_wgd.csv \\
        --wgd-spd ../human-data/dummy/dummy_human_wgd_spd.csv \\
        --le     ../human-data/dummy/dummy_human_le.csv \\
        --le-spd  ../human-data/dummy/dummy_human_le_spd.csv \\
        --out out/ --zip out/human_replication.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import human_io
import study2_lib as lib

ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = ANALYSIS_DIR.parent
WGD_LABELS = ROOT_DIR / "domains" / "WGD" / "Data" / "labels.csv"


def run(wgd=None, wgd_spd=None, le=None, le_spd=None,
        out_dir="out", results_root=None, benchmark_root=None,
        zip_path=None) -> list[Path]:
    """Generate every figure + the summary table into out_dir. Returns file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_kw = {}
    if results_root is not None:
        load_kw["results_root"] = results_root
    if benchmark_root is not None:
        load_kw["benchmark_root"] = benchmark_root

    def load(domain, human):
        return human_io.load_domain(domain, human_csv=human, **load_kw)

    lib.apply_style()

    # Load models (+ humans where provided). MedEval is always models-only.
    dce = {
        "WGD": load("WGD", wgd),
        "LifeEval": load("LifeEval", le),
        "MedEval": load("MedEval", None),
    }
    spd = {
        "WGD": load("WGD_SPD", wgd_spd),
        "LifeEval": load("LifeEval_SPD", le_spd),
        "MedEval": load("MedEval_SPD", None),
    }

    include_human = any(x is not None for x in (wgd, wgd_spd, le, le_spd))
    series = lib.build_series(include_human=include_human)

    outputs: list[Path] = []
    outputs.append(lib.fig_calibration(dce, series, out_dir, spd=False))
    outputs.append(lib.fig_calibration(spd, series, out_dir, spd=True))
    outputs.append(lib.fig_overconfidence_by_difficulty(dce, series, out_dir))
    outputs.append(lib.fig_overconfidence_by_percentile(dce, series, out_dir, spd=False))
    outputs.append(lib.fig_overconfidence_by_percentile(spd, series, out_dir, spd=True))
    outputs.append(lib.fig_ece_spd_improvement(dce, spd, series, out_dir))
    outputs.append(lib.fig_sex_bias_wgd(dce["WGD"], WGD_LABELS, series, out_dir))
    outputs.extend(lib.fig_wgd_demographics(WGD_LABELS, out_dir))

    # Summary table in three formats: LaTeX (paper-ready), CSV, PNG (readable).
    summary = lib.build_summary(dce, spd, series)
    tex_path = out_dir / "summary_table.tex"
    tex_path.write_text(lib.summary_to_latex(summary))
    outputs.append(tex_path)
    csv_path = out_dir / "summary_table.csv"
    summary.to_csv(csv_path, index=False)
    outputs.append(csv_path)
    outputs.append(lib.summary_to_png(summary, out_dir / "summary_table.png"))

    if zip_path is not None:
        zip_path = Path(zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in outputs:
                zf.write(p, arcname=p.name)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wgd", type=Path, help="Human WGD (direct) CSV.")
    parser.add_argument("--wgd-spd", type=Path, help="Human WGD SPD CSV.")
    parser.add_argument("--le", type=Path, help="Human LifeEval (direct) CSV.")
    parser.add_argument("--le-spd", type=Path, help="Human LifeEval SPD CSV.")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Output directory.")
    parser.add_argument("--zip", type=Path, help="Also bundle outputs into this zip.")
    args = parser.parse_args()

    outputs = run(
        wgd=args.wgd, wgd_spd=args.wgd_spd, le=args.le, le_spd=args.le_spd,
        out_dir=args.out, zip_path=args.zip,
    )
    print(f"Wrote {len(outputs)} files to {args.out}/")
    for p in outputs:
        print(f"  {p.name}")
    if args.zip:
        print(f"Zipped -> {args.zip}")


if __name__ == "__main__":
    main()
