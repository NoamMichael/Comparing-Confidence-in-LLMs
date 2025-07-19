import pandas as pd
import os

def read_first_prompt():
    """
    Opens each CSV file in the 'Prompts' folder, reads the 'Full Prompt' column,
    and prints the first prompt from that column.
    """
    prompts_dir = "Prompts/"
    if not os.path.isdir(prompts_dir):
        print(f"Directory not found: {prompts_dir}")
        return

    print(f"Reading prompts from: {os.path.abspath(prompts_dir)}\\n")

    for filename in sorted(os.listdir(prompts_dir)):
        if filename.endswith(".csv"):
            file_path = os.path.join(prompts_dir, filename)
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip')
                if 'Full Prompt' in df.columns and not df.empty:
                    first_prompt = df['Full Prompt'].iloc[0]
                    print(f"--- {filename} ---")
                    print(first_prompt)
                    print("\\n" + "="*80 + "\\n")
                else:
                    print(f"--- {filename} ---")
                    print("Could not find 'Full Prompt' column or file is empty.")
                    print("\\n" + "="*80 + "\\n")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    read_first_prompt()