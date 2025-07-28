import os
import argparse
import pandas as pd

# Model pricing data (cost per 1M tokens)
MODEL_PRICING = {
    "gpt-4o": {"input": 1.25, "output": 5.00, "batch_discount": 0.0},
    "gpt-03": {"input": 1.00, "output": 4.00, "batch_discount": 0.0},
    "Claude-3.7 Sonnet": {"input": 1.5, "output": 7.50, "batch_discount": 0.0},
    "Claude 3 Haiku": {"input": 0.125, "output": 0.625, "batch_discount": 0.0},
    "Claude-4 Sonnet": {"input": 1.5, "output": 7.50, "batch_discount": 0.0},
    "Gemini 1.5 Flash": {"input": 0.15, "output": 0.60, "batch_discount": 0.0},
    "Gemini 2.5 Pro": {"input": 1.25, "output": 10.00, "batch_discount": 0.0},
    "Deepseek V3": {"input": 0.135, "output": 0.550, "batch_discount": 0.0},
    "Deepseek R1": {"input": 0.135, "output": 0.550, "batch_discount": 0.0},
}

def estimate_token_count(text):
    """
    Estimates the number of tokens in a given text.
    Note: This is a simple approximation. For more accurate results, use a
    model-specific tokenizer like tiktoken.
    """
    return len(text.split())

def calculate_cost(input_tokens, output_tokens, model_pricing):
    """
    Calculates the cost based on input and output tokens and model pricing.
    """
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    total_cost = input_cost + output_cost
    
    # Apply batch discount
    total_cost *= (1 - model_pricing["batch_discount"])
    
    return total_cost

def main():
    parser = argparse.ArgumentParser(description="Estimate cost of running models on datasets.")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum number of tokens for the output.")
    
    args = parser.parse_args()

    prompts_dir = "Prompts"
    all_results = []

    for filename in os.listdir(prompts_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(prompts_dir, filename)
            try:
                df = pd.read_csv(filepath)
                # Assuming the prompt is in a column named 'prompt' or the first column
                if 'prompt' in df.columns:
                    text_content = " ".join(df['prompt'].astype(str))
                else:
                    text_content = " ".join(df.iloc[:, 0].astype(str))
                
                num_prompts = len(df)
                input_tokens = estimate_token_count(text_content)
                
                for model_name, pricing in MODEL_PRICING.items():
                    output_tokens = num_prompts * args.max_tokens
                    cost = calculate_cost(input_tokens, output_tokens, pricing)
                    all_results.append({
                        "Dataset": filename,
                        "Model": model_name,
                        "Input Tokens": input_tokens,
                        "Output Tokens (estimated)": output_tokens,
                        "Estimated Cost ($)": cost
                    })
            except Exception as e:
                print(f"Could not process {filename}: {e}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string())
        
        total_cost = results_df["Estimated Cost ($)"].sum()
        print(f"\nTotal Estimated Cost for all datasets and models: ${total_cost:.4f}")

if __name__ == "__main__":
    main()