import json
import sys
from pathlib import Path

def json_to_jsonl(input_path, output_path=None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".jsonl")

    # Read JSON file
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If the JSON is a single object, wrap it in a list
    if isinstance(data, dict):
        data = [data]

    # Write out line-delimited JSON
    with open(output_path, "w", encoding="utf-8") as f:
        for key, record in data[0].items():

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {input_path} → {output_path}")


if __name__ == "__main__":

    inp = 'Raw Results\\Llama\\Meta-Llama-3.1-8B-Instruct\\sciq_test_Meta-Llama-3.1-8B-Instruct.json'
    out = inp.replace('json', 'jsonl')

    json_to_jsonl(inp, out)
