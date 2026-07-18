# Human replication of Study 2

Regenerate the Study-2 figures and summary table with **humans treated as a
fifth "model"**, plotted in a distinct style (black diamonds / bars) next to the
four LLMs. The LLM results are already in `results/`; you supply the human data.

## Inputs

Up to four CSVs, all optional — a domain you omit is just drawn models-only:

| Flag        | Task                        | Results it stands in for      |
|-------------|-----------------------------|-------------------------------|
| `--wgd`     | Weight guessing, direct     | `results/WGD/`                |
| `--wgd-spd` | Weight guessing, SPD        | `results/WGD_SPD/`            |
| `--le`      | LifeEval, direct            | `results/LifeEval/`           |
| `--le-spd`  | LifeEval, SPD               | `results/LifeEval_SPD/`       |

Each file needs at least `question_id`, `Answer`, `Confidence`. Missing
ground-truth columns (`within_lbs`, `true_weight`, `min_age`, `sex`, `radius`,
…) are back-filled from the domain benchmark by `question_id`, so raw human
answers are enough. MedEval has no human counterpart and always stays LLM-only.

## Run

```bash
cd analysis
python human_replication.py \
    --wgd human_wgd.csv --wgd-spd human_wgd_spd.csv \
    --le  human_le.csv  --le-spd  human_le_spd.csv \
    --out out/ --zip out/human_replication.zip
```

Outputs (into `--out`): `calibration_combined.png`, `calibration_spd_combined.png`,
`overconfidence_by_difficulty.png`, `rq2_overconfidence_by_percentile.png`,
`rq3_overconfidence_by_percentile_spd.png`, `rq3_ece_spd_improvement.png`,
`posthoc_sex_bias_wgd.png`, the WGD demographics figures, and the summary table
as `summary_table.{tex,csv,png}`.

## Dummy data for testing

`make_dummy_human.py` fabricates stand-in human files by sampling one model
answer per question, so you can exercise the pipeline before real data exists:

```bash
python make_dummy_human.py            # writes ../human-data/dummy/dummy_human_*.csv
python human_replication.py \
    --wgd ../human-data/dummy/dummy_human_wgd.csv \
    --wgd-spd ../human-data/dummy/dummy_human_wgd_spd.csv \
    --le  ../human-data/dummy/dummy_human_le.csv \
    --le-spd  ../human-data/dummy/dummy_human_le_spd.csv \
    --out /tmp/llmc_out
```

## Modules

- `human_io.py` — loads models + optional humans into one scored frame; reproduces
  the notebook's scoring (plain `to_numeric`) so LLM numbers match the paper.
- `study2_lib.py` — shared config, calibration/ECE metrics, and figure/table builders.
- `human_replication.py` — the CLI that wires them together.

The same code is vendored into the website repo (`api/llmc_assets/`) to power the
upload tool at **noammichael.com/llmc**.
