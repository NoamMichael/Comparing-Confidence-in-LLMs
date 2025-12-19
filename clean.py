import pandas as pd
import numpy as np
from pathlib import Path


# Global Variables
MCQ_QSETS = ['LSAT-AR', 'SAT-EN', 'SciQ']
SC_COLS = ["Stated Confidence A", "Stated Confidence B", "Stated Confidence C","Stated Confidence D","Stated Confidence E"]
MODEL_NAMES = {
    "gpt-4o": "GPT-4o",
    "o3-2025-04-16": "GPT-o3",
    "claude-sonnet-4-20250514": "Claude-Sonnet-4",
    "claude-3-7-sonnet-20250219": "Claude-Sonnet-3.7",
    "claude-3-haiku-20240307": "Claude Haiku 3",
    "gemini-2.5-pro": "Gemini-2.5-Pro",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Meta-Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "deepseek-r1": "DeepSeek-R1",
    "deepseek-v3": "DeepSeek-V3",
}
VERBOSE = True


def drop_bad_qid(data: pd.DataFrame):
    """
    This function looks at all QIDs which don't show up for all models. Some responses failed to excute.
    """
    df = data.copy()
    num_models = len(df['Model'].unique())
    if VERBOSE: print(f"    Number of models: {num_models}")

    s = df['combined_name'].value_counts() % num_models 
    bad_qid = s.index[s.ne(0)].tolist()  # List of bad_qids

    df = df[~df['combined_name'].isin(bad_qid)]
    return df

def drop_uncoerced(data: pd.DataFrame):
    df = data.copy()
    # this gets the qid and qset where coerce is false
    bad_qid_qset = df[
        (df["Coerce"] == False) | 
        (df["Stated Confidence Answer"].astype(str).str.isnumeric())
        ][["Question ID", "Question Set"]]
    bad = set(bad_qid_qset["Question ID"] +  "_" + bad_qid_qset["Question Set"])

    mask = ~df["combined_name"].isin(bad)   # True = keep

    df = df[mask]

    return df
    
def clean_mcq(data: pd.DataFrame):

    df = data.copy()
    ## Clean up MCQ Question Sets

    cc = df[df['Question Set'].isin(MCQ_QSETS)]
    cc_letters = cc[["Stated Confidence A", "Stated Confidence B", "Stated Confidence C","Stated Confidence D","Stated Confidence E"]].copy()
    sum_confidence = cc_letters.sum(axis = 1)

    # Sum responses had all 0's for the stated confidence. We drop these rows as it doesn't make sense to include them in our analysis.
    con_mask = cc[sum_confidence == 0.0]["combined_name"]

    # Indixes that aren't in con_mask
    df = df.loc[~df["combined_name"].isin(con_mask)]

    return df

def clean_LifeEval(data: pd.DataFrame):
    df = data.copy()
    le_df = df[df["Question Set"] == "LifeEval"]

    con_isnum = pd.to_numeric(le_df['Stated Confidence Answer'], errors='coerce').notna()
    le_bad_qid= le_df[~con_isnum]["combined_name"]


    df = df[(~df["combined_name"].isin(le_bad_qid))]

    return df

def normalize_columns(df, column_list):
    """
    Normalizes specified columns in a DataFrame so that their values 
    sum to 1.0 across each row.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column_list (list): A list of column names (strings) to normalize.
        
    Returns:
        pd.DataFrame: The DataFrame with the specified columns normalized.
    """
    # 1. Calculate the sum across the specified columns for each row (axis=1)
    # This creates a Series where each value is the sum for that row.
    row_sums = df[column_list].sum(axis=1)
    
    # 2. Prevent division by zero: replace any zero sums with 1 
    # (this ensures division by 1 for rows that summed to 0, leaving them unchanged)
    row_sums_safe = row_sums.replace(0, 1)
    
    # 3. Divide the original columns by the row_sums_safe Series.
    # We use axis=0 to align the Series (row_sums_safe) with the DataFrame rows.
    df[column_list] = df[column_list].div(row_sums_safe, axis=0)

    df['Stated Confidence Answer (MCQ)'] = df['Stated Confidence Answer (MCQ)'] / row_sums_safe
    
    return df

def main():
    # 1. Import Raw Results df
    results_path = Path(r"Combined Results\combined_raw.csv")
    combined_df = pd.read_csv(results_path)
    combined_clean = combined_df.copy()

    # Add 'combined_name' column
    combined_clean['combined_name'] = combined_clean['Question ID'] + "_" + combined_clean["Question Set"]

    # Drop Question IDs which don't appear for all models
    combined_clean = drop_bad_qid(combined_clean)


    # Drop rows where 'Coerce' is false
    combined_clean = drop_uncoerced(combined_clean)

    # Drop rows where the MCQ question summed to zero
    combined_clean = clean_mcq(combined_clean)

    # Drop rows of LifeEval where the answer was not able to be converted to a float
    combined_clean = clean_LifeEval(combined_clean)

    # Normalize stated confidence values to sum to 1
    combined_clean = normalize_columns(combined_clean, SC_COLS)
    combined_clean['Model'] = combined_clean['Model'].map(MODEL_NAMES)


    
    print("Finished Cleaning:")
    clean_path = Path("Combined Results/combined_clean.csv")
    combined_clean.to_csv(clean_path, index=False, encoding="utf-8")
    print(f"Saved cleaned DataFrame to: {clean_path}")
    # print(combined_clean.iloc[0])


if __name__ == "__main__":
    main()
