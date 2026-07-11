"""One-shot cleanup of the study-1 notebooks after LifeEval was archived.

LifeEval lives in `archive/lifeeval/` (study 1) and is actively developed in
`study-2-bayeseval/`. The notebooks still carried LifeEval-only cells (some of
which hard-error because their input files moved to the archive) plus
Windows-style backslash paths that break on Linux/macOS. This script:

  1. edits shared cells in place (grading branches, qset dicts, plot orders),
  2. deletes LifeEval-only cells,
  3. normalizes backslash path literals to forward slashes,
  4. clears stale outputs of every cell it touched.

Idempotent: exits without writing if a notebook has already been stripped.
Run from the study-1 root:  python scripts/strip_lifeeval_from_notebooks.py
"""

import ast
import json
import re
import sys
from pathlib import Path

ANALYSIS = Path("analysis.ipynb")
WORKFLOW = Path("Workflow/get_results_analysis.ipynb")

# Identifiers that only ever existed for LifeEval; any cell still referencing
# one of them after the in-place edits belongs to the archived analysis.
ORPHAN_IDENTIFIERS = [
    "score_life_eval",
    "compute_prob",
    "qid_to_rads",
    "get_age(",
    "compute_entropy",
    "best_df",
    "le_summary",
    "best_results_le",
]


def src_of(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def edit(cell, old, new, must_exist=True):
    text = src_of(cell)
    if old not in text:
        if must_exist:
            raise SystemExit(f"Expected snippet not found:\n{old[:120]}...")
        return False
    set_src(cell, text.replace(old, new))
    return True


def find_cell(nb, snippet, cell_type=None):
    hits = [
        c
        for c in nb["cells"]
        if snippet in src_of(c) and (cell_type is None or c["cell_type"] == cell_type)
    ]
    if len(hits) != 1:
        raise SystemExit(f"Expected exactly 1 cell containing {snippet!r}, found {len(hits)}")
    return hits[0]


def remove_block(cell, start_marker, end_marker, exact=False):
    """Remove lines from the one containing start_marker up to (excluding) the
    one containing end_marker. With exact, both markers must equal the whole
    line (ignoring the trailing newline) so indentation disambiguates."""
    lines = src_of(cell).splitlines(keepends=True)

    def hits(marker, after=-1):
        if exact:
            return [i for i, l in enumerate(lines) if i > after and l.rstrip("\n") == marker]
        return [i for i, l in enumerate(lines) if i > after and marker in l]

    starts = hits(start_marker)
    if len(starts) != 1:
        raise SystemExit(f"Expected 1 line matching {start_marker!r}, found {len(starts)}")
    start = starts[0]
    ends = hits(end_marker, after=start)
    if not ends:
        raise SystemExit(f"No end marker {end_marker!r} after {start_marker!r}")
    set_src(cell, "".join(lines[:start] + lines[ends[0]:]))


PATH_DIRS = (
    "Formatted Benchmarks",
    "Combined Results",
    "Parsed Results",
    "Raw Results",
    "Plots",
    "MISC",
)


def normalize_paths(nb):
    pattern = re.compile(r"(" + "|".join(PATH_DIRS) + r")\\+")
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = src_of(cell)
        # also collapse backslashes between already-normalized segments,
        # e.g. Plots\Summary Plots\foo.png -> Plots/Summary Plots/foo.png
        new = pattern.sub(r"\1/", text)
        prev = None
        while prev != new:
            prev = new
            new = re.sub(r"(Plots/[^\"'\n]*?)\\+", r"\1/", new)
        if new != text:
            set_src(cell, new)
            n += 1
    return n


def keeps_only_filter(text):
    """True if every LifeEval mention is a `!= "LifeEval"` row filter."""
    residue = text.replace('!= "LifeEval"', "").replace("!= 'LifeEval'", "")
    return "LifeEval" not in residue and "life_eval" not in residue


def strip_analysis(nb):
    cells = nb["cells"]

    # --- in-place edits -----------------------------------------------------
    edit(
        find_cell(nb, 'elif qset_name == "LifeEval":'),
        '    elif qset_name == "LifeEval":\n'
        '        df["Question ID"]= df["Question ID"].astype(int)\n'
        '        df["Score"] = score_life_eval(df, gold_df)\n'
        '        df["Question ID"]=df["Question ID"].astype(str)\n',
        "",
    )

    c12 = find_cell(nb, "gold_paths = {")
    edit(c12, " 'life_eval': \"LifeEval\",\n", "", must_exist=True)
    edit(c12, " 'life_eval': \"LifeEval\",\n", "", must_exist=False)  # second copy
    edit(
        c12,
        '    "LifeEval":    r"Formatted Benchmarks\\PeriodLifeTable_2022_RawData.csv",\n',
        "",
    )

    note = find_cell(nb, "**LifeEval filter.**", cell_type="markdown")
    set_src(
        note,
        "**LifeEval filter.** LifeEval is archived (see `archive/lifeeval/`) and actively\n"
        "developed in `study-2-bayeseval/`. The canonical `combined_raw.csv` keeps its rows\n"
        "as the preregistered record, so we drop them here. All LifeEval-specific analysis\n"
        "cells have been removed from this notebook; see the archive for that record.\n",
    )

    for cell in cells:
        text = src_of(cell)
        if 'qset_order = ["BoolQ", "HaluEval", "LifeEval", "LSAT", "SAT", "SciQ"]' in text:
            edit(
                cell,
                'qset_order = ["BoolQ", "HaluEval", "LifeEval", "LSAT", "SAT", "SciQ"]',
                'qset_order = ["BoolQ", "HaluEval", "LSAT", "SAT", "SciQ"]',
            )
        if "no_le = d[d['Question Set'] != 'LifeEval']\n" in src_of(cell):
            edit(cell, "no_le = d[d['Question Set'] != 'LifeEval']\n", "")
        if '    "Probability Estimation": [\'LifeEval\']\n' in src_of(cell):
            edit(cell, '    "Probability Estimation": [\'LifeEval\']\n', "")
        elif '    "Probability Estimation": [\'LifeEval\'],\n' in src_of(cell):
            edit(cell, '    "Probability Estimation": [\'LifeEval\'],\n', "")

    # now-unreachable else branch (Probability Estimation) in the aggregate cells
    for cell in cells:
        if "qset_types" in src_of(cell):
            edit(
                cell,
                "    else:\n"
                "        scores = subset['Score']\n"
                "        conf = subset['Stated Confidence Answer']\n",
                "",
                must_exist=False,
            )

    # get_tpa: LifeEval had a dedicated token-prob column; without it TP is
    # simply the row max over the option/True/False token probabilities.
    tpa = find_cell(nb, "def get_tpa")
    edit(
        tpa,
        '    # 3. Use np.where (Condition, Value_if_True, Value_if_False)\n'
        '    tpa = np.where(\n'
        '        df["Question Set"] == "LifeEval", \n'
        '        df["Token Probability Answer"], \n'
        '        row_maxes\n'
        '    )\n'
        '    \n'
        '    return pd.Series(tpa, index=df.index)\n',
        "    return pd.Series(row_maxes, index=df.index)\n",
    )
    edit(
        tpa,
        '#tokens["Token Probability Answer 2"] = tokens.apply(lambda x: x[f"Token Probability {x["Answer"].strip()}"] if x["Question Set"] != "LifeEval" else x["Token Probability Answer"], axis = 1)\n',
        "",
        must_exist=False,
    )

    # Rounding cell: keep percent_rounded / cc_r / cc_c (used by the summary
    # tables below), drop the LifeEval-only slices and prints.
    rounding = find_cell(nb, "def percent_rounded")
    set_src(
        rounding,
        "reasoning_models = [\n"
        "    'Claude-Sonnet-3.7',\n"
        "    'Claude-Sonnet-4',\n"
        "    'DeepSeek-R1',\n"
        "    'Gemini-2.5-Pro',\n"
        "    'GPT-o3',\n"
        "]\n"
        "\n"
        "def percent_rounded(df, col: str = 'Stated Confidence Answer'):\n"
        "    return np.mean((df[col].astype(float) * 100) % 5 == 0)\n"
        "\n"
        "cc_r = combined_clean[combined_clean['Model'].isin(reasoning_models)]\n"
        "cc_c = combined_clean[~combined_clean['Model'].isin(reasoning_models)]\n"
        "\n"
        "print('Stated Confidence:')\n"
        "print(f'    Reasoning Models: {percent_rounded(cc_r) * 100:.5}% rounded')\n"
        "print(f'    Chat Models:      {percent_rounded(cc_c) * 100:.5}% rounded')\n"
        "print('Token Probability:   # Note: we don\\u2019t get Token Probs for Reasoning Models')\n"
        "print(f'    Reasoning Models: {percent_rounded(cc_r, \"Token Probability Answer\") * 100:.5}% rounded')\n"
        "print(f'    Chat Models:      {percent_rounded(cc_c, \"Token Probability Answer\") * 100:.5}% rounded')\n",
    )

    heading = find_cell(nb, "## Pearson Coef. By Reasoning v. Not Reasoning Models", cell_type="markdown")
    set_src(heading, "## Confidence Rounding: Reasoning v. Chat Models\n")

    # --- cell deletions ------------------------------------------------------
    md_delete_sigs = [
        "# LifeEval Analysis",
        "## Difficulty by QID",
        "**NOTE** Include in supplement",
        "**Lineplot**",
        "**Scatterplot**",
        "## LifeEval Calibration Plot",
        "Lets clean it up to include what we want",
        "## Overconfidence by Radii",
        "### Stated Overconfidence",
        "**NOTE** Include in paper.",
        "### Token Overconfidence",
        "Hard-Esay effect is prevelent",
        "## LifeEval Summary Table",
        "Gut check for LifeEval",
    ]

    def should_delete(cell):
        text = src_of(cell)
        if cell["cell_type"] == "markdown":
            return any(sig in text for sig in md_delete_sigs)
        if ("LifeEval" in text or "life_eval" in text) and not keeps_only_filter(text):
            return True
        if any(ident in text for ident in ORPHAN_IDENTIFIERS):
            return True
        if text.strip().startswith("rad_20 = le["):
            return True
        if text.strip() == "%pip install scipy":
            return True
        return False

    deleted = [i for i, c in enumerate(cells) if should_delete(c)]
    nb["cells"] = [c for i, c in enumerate(cells) if i not in set(deleted)]
    return deleted


def strip_workflow(nb):
    deleted = []

    # qsets dicts (parsing cell + plotting cell)
    for cell in nb["cells"]:
        text = src_of(cell)
        for variant in (
            '      "life_eval": "LifeEval",\n',
            '    "life_eval": "LifeEval", \n',
            '    "life_eval": "LifeEval",\n',
        ):
            if variant in text:
                edit(cell, variant, "")
                text = src_of(cell)

    # Parsing cell: token-prob LifeEval branch + coerce condition
    parse = find_cell(nb, "elif qset_display == 'LifeEval':")
    remove_block(parse, "                elif qset_display == 'LifeEval':", "                else:", exact=True)
    edit(
        parse,
        "if (qset_display != 'LifeEval') and (qset_display != 'HaluEval'):",
        "if qset_display != 'HaluEval':",
    )

    # Function-definitions cell: drop the four LifeEval-only helpers
    funcs = find_cell(nb, "def make_summary_plots")
    remove_block(funcs, "def qid_to_rads", "def get_gini")

    # Parser-definitions cell has its own dead copy of qid_to_rads
    parser_defs = find_cell(nb, "def quick_parse")
    remove_block(parser_defs, "def qid_to_rads", "def field_probs")

    # Make-All-Plots cell: valid-ID branch, scoring branch, token branch
    plots = find_cell(nb, "## Make All Plots")
    remove_block(plots, "    elif qset_name == 'LifeEval':", "    # --- Convert DFs to only usable IDs ---", exact=True)
    remove_block(plots, "            elif qset_name == 'LifeEval':", "            elif qset_name == 'LSAT-AR':", exact=True)
    remove_block(plots, "                    if qset_name == 'LifeEval':", "                    elif qset_name == 'BoolQ':", exact=True)
    edit(plots, "                    elif qset_name == 'BoolQ':", "                    if qset_name == 'BoolQ':")
    edit(plots, " #TODO --- Fix this for LifeEval", "", must_exist=False)

    # Stated-vs-token ECE scatter: LifeEval marker entry
    edit(find_cell(nb, 'dataset_markers = {'), '    "LifeEval": "s",\n', "")

    # Commented LifeEval fragment in the ECE table cell
    ece = find_cell(nb, 'only_ece = table_df')
    edit(
        ece,
        'only_ece#.loc[only_ece.index.get_level_values("Dataset") == "LifeEval"]',
        "only_ece",
    )

    return deleted


def verify(nb, name):
    bad = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = src_of(cell)
        if ("LifeEval" in text or "life_eval" in text) and not keeps_only_filter(text):
            bad.append((name, i, "LifeEval reference"))
        if re.search(r'r?["\'](?:' + "|".join(PATH_DIRS) + r")\\", text):
            bad.append((name, i, "backslash path"))
        try:
            # magics can't be parsed; skip cells that use them
            if not any(l.lstrip().startswith(("%", "!")) for l in text.splitlines()):
                ast.parse(text)
        except SyntaxError as e:
            bad.append((name, i, f"syntax error: {e}"))
    return bad


def main():
    changed = False
    for path, strip in ((ANALYSIS, strip_analysis), (WORKFLOW, strip_workflow)):
        nb = json.loads(path.read_text())
        already = not any(
            ("LifeEval" in src_of(c) or "life_eval" in src_of(c)) and not keeps_only_filter(src_of(c))
            for c in nb["cells"]
            if c["cell_type"] == "code"
        )
        if already:
            print(f"{path}: already stripped, skipping")
            continue
        before = len(nb["cells"])
        deleted = strip(nb)
        n_paths = normalize_paths(nb)
        problems = verify(nb, str(path))
        if problems:
            for p in problems:
                print("PROBLEM:", p)
            raise SystemExit(f"{path}: verification failed, not writing")
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(
            f"{path}: {before} -> {len(nb['cells'])} cells "
            f"({before - len(nb['cells'])} deleted), {n_paths} cells path-normalized"
        )
        changed = True
    if not changed:
        print("Nothing to do.")


if __name__ == "__main__":
    main()
