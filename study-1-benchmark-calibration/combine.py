# Initialize
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from pathlib import Path
import json

# Global Variables:
FOLDER_PATH = r"Parsed Results"
MCQ_QSETS = ['LSAT-AR', 'SAT-EN', 'SciQ']
SAVE_FOLDER_PATH = r"Combined Results"

QSET_RENAME = {
 'boolq_valid': "BoolQ",
 'halu_eval_qa': "HaluEval",
 'lsat_ar_test': "LSAT-AR",
 'sat_en': "SAT-EN",
 'sciq_test':"SciQ"
}

GOLD_PATHS = {
    "BoolQ": "Formatted Benchmarks/boolq_valid_formatted.csv",
    "HaluEval": "Formatted Benchmarks/halu_eval_qa_formatted.csv",
    "LSAT-AR": "Formatted Benchmarks/lsat_ar_test_formatted.csv",
    "SAT-EN": "Formatted Benchmarks/sat_en_formatted.csv",
    "SciQ": "Formatted Benchmarks/sciq_test_formatted.csv"
}


def folder_tree_dict(root, *, include_files=True, follow_symlinks=False, ignore_hidden=True):
    root = Path(root)

    def build(p: Path):
        out = {}
        for entry in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if ignore_hidden and entry.name.startswith("."):
                continue
            try:
                if entry.is_dir() and (follow_symlinks or not entry.is_symlink()):
                    out[entry.name] = build(entry)
                else:
                    if include_files:
                        out[entry.name] = None  # or {"size": entry.stat().st_size}
            except PermissionError:
                out[entry.name] = "<permission-denied>"
        return out

    return {root.name: build(root)}

def grade_df(source_df, gold_df, qset_name):
    df = source_df.copy()

    if qset_name in MCQ_QSETS:
        # Make sure the QIDs are in the right order
        df["Question ID"] = df["Question ID"].astype(int) 
        df = df.sort_values(by="Question ID", ascending=True).reset_index()
        df["Question ID"] = df["Question ID"].astype(str)  ## make sure back as str for downstream tasks

        gold_df["Question ID"] = gold_df["Question ID"].astype(str)
        temp = pd.merge(source_df, gold_df, on = "Question ID")
        scores = (temp["Answer"].str.lower().str.strip() == temp["Correct Answer Letter"].str.lower().str.strip()).astype(float)
        df["Score"] = scores
        df['Correct Answer'] = temp["Correct Answer Letter"].str.upper().str.strip()

        # assumes stated confidences live in columns ["A","B","C","D"]
        if qset_name == "LSAT-AR":
            opt_cols = ["A","B","C","D", "E"]
            # normalize answer letters
            ans = df["Answer"].astype("string").str.strip().str.upper()

            # map letters to column indices
            idx = ans.map({"A":0, "B":1, "C":2, "D":3, "E":4 }).to_numpy()
        else:
            opt_cols = ["A","B","C","D"]
            # normalize answer letters
            ans = df["Answer"].astype("string").str.strip().str.upper()

            # map letters to column indices
            idx = ans.map({"A":0, "B":1, "C":2, "D":3}).to_numpy()

        vals = df[opt_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        mask = ~np.isnan(idx)

        chosen = np.full(len(df), np.nan, dtype=float)
        chosen[mask] = vals[mask, idx[mask].astype(int)]
        df["Stated Confidence Answer (MCQ)"] = chosen


    elif qset_name == "BoolQ":
        gold_df["Question ID"] = gold_df["Question ID"].astype(str)
        temp = pd.merge(source_df, gold_df, on = "Question ID")

        bscores = (temp["Answer"].astype(str) == temp["Correct Answer"].astype(str)).astype(float)
        df["Score"] = bscores
        df['Correct Answer'] = temp["Correct Answer"].astype(str)
        
    elif qset_name == "HaluEval":
        df["Score"] = df["Question ID"].str.contains("_r").astype(float)
    else:
        df["Score"] = "UNRECOGNIZED QUESTION SET"
    return df





if __name__ == "__main__":
    print(f"{"%" * 64}\nCombining All Results from {FOLDER_PATH}\n{"%" * 64}")
    combined_df = pd.DataFrame()
    full_folder_path = Path(FOLDER_PATH)

    folder_abstraction_dict = folder_tree_dict(full_folder_path)[FOLDER_PATH]
    
    print(f"Models: {[model for model in folder_abstraction_dict.keys()]}")

    print("%" * 64  + "\nProcessing Results:")

    for model_type, models in folder_abstraction_dict.items():
        model_type_path = full_folder_path / model_type
        for model_name, qsets in models.items():
            model_path = model_type_path / model_name
            print(f"    MODEL: {model_name}")

            for qset_file_name in qsets:
                splitter = f"_{model_name}"
                qset_name = qset_file_name.split(splitter)[0]
                qset_path = model_path / qset_file_name
                source_df = pd.read_csv(qset_path)

                print(f"        {qset_name}    ")

                source_df["Model"] = model_name
                source_df["Model Type"] = model_type

                qset_display = QSET_RENAME[qset_name]
                source_df["Question Set"] = qset_display
                source_df["Question ID"] = source_df["Question ID"].astype(str)

                gold_df_path = GOLD_PATHS[qset_display]

                gold_df = pd.read_csv(gold_df_path)
                
                scored_qset = grade_df(source_df = source_df, gold_df = gold_df, qset_name= qset_display)

                combined_df = pd.concat([combined_df, scored_qset], ignore_index=True)

    print(f"{"%" * 64}\nSuccessfully combined all files from '{FOLDER_PATH}' into one CSV!\n{"%" * 64}")


    combined_df.drop(["Unnamed: 0", "Question ID.1"], axis = 1, inplace = True, errors = "ignore")

    col_rename_map ={
    # Metadata
    'Question Set': "Question Set",
    'Question ID': "Question ID",
    'Model': "Model",
    'Model Type': "Model Type",
    'coerce': "Coerce",

    # Model Response

    'content': "Content",
    'Reasoning': "Reasoning",
    'Answer': "Answer",

    # Stated Confidence
    'Confidence': "Stated Confidence Answer",
    "A": "Stated Confidence A",
    "B": "Stated Confidence B",
    'C': "Stated Confidence C",
    'D': "Stated Confidence D",
    'E': "Stated Confidence E",

    # Token Probability
    'True_prob': "Token Probability True",
    'False_prob': "Token Probability False",
    'Answer_prob': "Token Probability Answer",
    'A_prob': "Token Probability A",
    'B_prob': "Token Probability B",
    'C_prob': "Token Probability C",
    'D_prob': "Token Probability D",
    'E_prob': "Token Probability E"
    }

    combined_df = combined_df.rename(columns = col_rename_map)

    raw_path = Path("Combined Results/combined_raw.csv")
    combined_df.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"Successfully saved to {raw_path}")
