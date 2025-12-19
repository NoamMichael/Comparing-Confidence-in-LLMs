# Initialize
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from pathlib import Path
import math as m
import json

# Global Variables:
FOLDER_PATH = r"Parsed Results"
MCQ_QSETS = ['LSAT-AR', 'SAT-EN', 'SciQ']
SAVE_FOLDER_PATH = r"Combined Results"

QSET_RENAME = {
 'boolq_valid': "BoolQ",
 'halu_eval_qa': "HaluEval",
 'life_eval': "LifeEval",
 'lsat_ar_test': "LSAT-AR",
 'sat_en': "SAT-EN",
 'sciq_test':"SciQ"
}

GOLD_PATHS = {
    "BoolQ": r"Formatted Benchmarks\boolq_valid_formatted.csv",
    "HaluEval": r"Formatted Benchmarks\halu_eval_qa_formatted.csv",
    "LifeEval":    r"Formatted Benchmarks\PeriodLifeTable_2022_RawData.csv",
    "LSAT-AR": r"Formatted Benchmarks\lsat_ar_test_formatted.csv", 
    "SAT-EN": r"Formatted Benchmarks\sat_en_formatted.csv",
    "SciQ": r"Formatted Benchmarks\sciq_test_formatted.csv"    
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

def get_age(qid: int) -> int:
    if qid < 404:
        return m.floor(
            abs(
                (qid) / 4
                )
            )
    else:
        return m.floor(
            abs(
                (qid - 404) / 4
                )
            )

def compute_prob(point_estimate: float,
                 R: float,
                 gender: str,          # 'male' or 'female' (case-insensitive)
                 min_age: int,         # condition "already lived at least min_age"
                 df: pd.DataFrame) -> float:
    """
    Using a life table with columns:
      - 'Age'
      - 'Death probability (MALE)'
      - 'Death probability (FEMALE)'
    compute P(death occurs within [point_estimate - R, point_estimate + R] | survived to min_age).
    """

    if "Age" not in df.columns:
        raise ValueError("Expected an 'Age' column in the life table.")

    g = gender.strip().lower()
    if g not in ("male", "female"):
        raise ValueError("gender must be 'male' or 'female'")

    qx_col_map = {
        "male":   "Death probability (MALE)",
        "female": "Death probability (FEMALE)",
    }
    qx_col = qx_col_map[g]
    if qx_col not in df.columns:
        raise ValueError(f"Expected column '{qx_col}' in the life table.")


    # Work on a clean copy sorted by Age
    global tab
    tab = df[["Age", qx_col]].copy().sort_values("Age").reset_index(drop=True).dropna()
    tab.rename(columns={qx_col: "q"}, inplace=True)
    tab["Age"] = tab["Age"].astype(int)
    tab["q"] = tab["q"].astype(float)

    # Integer-age window [lo, hi), clamped to table bounds and min_age
    table_min = int(tab["Age"].min())
    table_max = int(tab["Age"].max())  # last age with q_x for [x, x+1)
    lo = max(int(np.floor(point_estimate - R)), int(min_age), table_min)
    hi = min(int(np.ceil(point_estimate + R)), table_max + 1)  # exclusive upper bound

    if hi <= lo:
        return 0.0

    # Align to contiguous ages and extract q_x as numpy
    ages = np.arange(table_min, table_max + 1, dtype=int)
    sub = tab.set_index("Age").reindex(ages)
    if sub["q"].isna().any():
        # restrict to contiguous valid block if necessary
        valid = sub["q"].notna()
        first = int(ages[valid.argmax()])
        last = int(ages[::-1][valid.iloc[::-1].argmax()])
        lo = max(lo, first)
        hi = min(hi, last + 1)
        if hi <= lo:
            return 0.0
        sub = sub.loc[first:last]
        ages = sub.index.values

    q = sub["q"].to_numpy()
    offset = int(ages[0])

    # Survival from min_age (m)
    m = max(int(min_age), int(ages[0]))
    if m >= hi:
        return 0.0

    one_minus_q = 1.0 - q
    start = m - offset
    end_excl = hi - offset

    # Relative survival S_rel(x) = Π_{k=m}^{x-1} (1-q_k), with S_rel(m)=1
    S_rel = np.ones(end_excl - start + 1, dtype=float)
    if end_excl - start > 0:
        S_rel[1:] = np.cumprod(one_minus_q[start:end_excl])

    # Sum P(T in [x, x+1) | T >= m) = S_rel(x) * q_x for x = max(lo,m) .. hi-1
    x0 = max(lo, m)
    xs = np.arange(x0, hi, dtype=int)
    if xs.size == 0:
        return 0.0

    idx = xs - offset
    S_rel_x = S_rel[(xs - m)]
    q_x = q[idx]
    prob = float(np.sum(S_rel_x * q_x))

    # Clamp for numerical safety
    return max(0.0, min(1.0, prob))

def score_life_eval(df, act_table):
    answers = df['Answer']
    
    #confidence = df['Stated Confidence'].astype(float)
    qid = df['Question ID']

    # Get Radius
    radius_list = [1, 5, 10, 20]
    # Get the modulus of QID then use that as an index for radius_list such that 0-> 1, 1-> 5, 2-> 10, 3-> 20
    mod_qid = df['Question ID'].astype('int').apply(lambda x: x % 4)


    rads = qid_to_rads(qid)
    #df['radius'] = rads

    #Gold Answer:

    all_data = pd.DataFrame({
        'Question ID': qid,
        'Answer': answers,
        #'Confidence': confidence.astype(float),
        'Radius': rads,
    })

    data = all_data[all_data['Answer'].notna()].copy()
    data['Gender'] = ['female' if i >= 404 else 'male' for i in data.index]
    data['Age'] = [get_age(qid) for qid in data['Question ID']]


    data['Score'] = data.apply(lambda row: compute_prob(
        point_estimate= row['Answer'],
        min_age= row['Age'],
        gender= row['Gender'],
        R= row['Radius'],
        df = act_table
        ),
        axis = 1
        )


    #data['Overconfidence'] = (data['Confidence'] - data['Score'])
    return data["Score"]

def qid_to_rads(qid: pd.Series)-> pd.Series:
    radius_list = [1, 5, 10, 20]
    mod_qid = qid.astype('int').apply(lambda x: x % 4)
    rads = mod_qid.apply(lambda i: radius_list[i])
    return rads

def grade_df(source_df, gold_df, qset_name):
    df = source_df.copy()
    mask = df.index
    #print(f"            Length: {len(mask)}")
    
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
        # Make sure the QIDs are in the right order
        # df["Question ID"] = df["Question ID"].astype(int) 
        # df = df.sort_values(by="Question ID", ascending=True).reset_index()
        # df["Question ID"] = df["Question ID"].astype(str)  ## make sure back as str for downstream tasks

        gold_df["Question ID"] = gold_df["Question ID"].astype(str)
        temp = pd.merge(source_df, gold_df, on = "Question ID")

        bscores = (temp["Answer"].astype(str) == temp["Correct Answer"].astype(str)).astype(float)
        df["Score"] = bscores
        df['Correct Answer'] = temp["Correct Answer"].astype(str)
        
    elif qset_name == "LifeEval":
        df["Question ID"]= df["Question ID"].astype(int)
        df["Score"] = score_life_eval(df, gold_df)
        df["Question ID"]=df["Question ID"].astype(str)
    elif qset_name == "HaluEval":
        df["Score"] = df["Question ID"].str.contains("_r").astype(float)
    else:
        df["Score"] = "UNRECOGINIZED QUESTION SET"
    """
    Don wants: one column that indicates stated confidence in the correct answer, with remaining 
    in correct answers in different columns.
    (It would of course follow that token prob should be reorganized the same way.) 

    Currently we evaluate the whether the answer was correct and then the confidence it was assigned. Is this different?
    Don's Approach: Grade the question based off of the correct answer.
    My Approach: Grade the question based off of the chosen answer.
    """
    #print(f"            Length: {len(df)}")
    return df





if __name__ == "__main__":
    print(f"{"%" * 64}\nCombining All Results from {FOLDER_PATH}\n{"%" * 64}")
    combined_df = pd.DataFrame()
    # Get the absolute path to the script's directory
    script_dir = Path(__file__).parent
    # Get the project root by going up two levels (from Workflow/Analysis to the root)
    project_root = script_dir.parent.parent
    # Construct the full path to the Parsed Results folder
    full_folder_path = Path(FOLDER_PATH)
    print(full_folder_path)

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
                
                #--------- Write a function to spit out a dataframe w/ model_name, qset_name, true_answer and concat it

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

    ## Still need to add correct answer and score

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



    #combined_df["Question Set"] = combined_df["Question Set"].map(qset_rename)
    print(f"\nSample row:\n")
    print(combined_df.iloc[2048])
    raw_path   = Path("Combined Results/combined_raw.csv")
    combined_df.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"Successfully saved to {raw_path}")



        