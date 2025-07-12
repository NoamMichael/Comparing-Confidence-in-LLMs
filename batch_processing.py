import pandas as pd
import openai
import os
import anthropic
import google.generativeai as genai
from openai import OpenAIError
from abc import ABC, abstractmethod
from anthropic import Anthropic, AsyncAnthropic, APIError

from typing import Optional

models = []
datasets = []


class BatchModels(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run_batch(self, prompts: pd.Series, **kwargs):
        pass

    @abstractmethod
    def make_batch(self, prompts: pd.Series, **kwargs):
        pass

class GPTModels(BatchModels):
    def __init__(self, name: str):
        self.name = name
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
    def make_batch(prompts: pd.Series, model: str, system_prompt: str, logprobs=True, top_logprobs=5,
                  temperature=0, output_file="batch.jsonl", url="/v1/chat/completions"):
        """
        Creates a JSONL file for the OpenAI batch API from a series of prompts using the chat completions format.

        Args:
            prompts (pd.Series): A pandas Series of prompt strings.
            model (str): The name of the OpenAI model to use (e.g., "gpt-4").
            system_prompt (str): The shared system message for all prompts.
            logprobs (bool): Whether to request logprobs in the output.
            top_logprobs (int): Number of top logprobs to include.
            temperature (float): Sampling temperature for generation.
            output_file (str): File path for the resulting JSONL file.
            url (str): API endpoint URL, defaulting to the chat completions endpoint.
        """
        requests = []

        for idx, user_prompt in prompts.items():
            request = {
                "custom_id": str(idx),
                "method": "POST",
                "url": url,
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": 256  # adjust as needed
                }
            }

            # Only include logprobs fields if requested
            if logprobs:
                request["body"]["logprobs"] = top_logprobs

            requests.append(request)

        # Write each JSON object as a line in the JSONL file
        with open(output_file, "w", encoding="utf-8") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        print(f"Saved {len(requests)} prompts to {output_file}")
    def run_batch(self,
                  prompts: pd.Series,
                  system_prompt: str,
                  logprobs=True,
                  top_logprobs=5,
                  temperature=0,
                  output_file="batch.jsonl",
                  url="/v1/chat/completions",
                  endpoint_id=None):
        """
        Creates and submits a batch job to the OpenAI API using a series of prompts.

        Args:
            prompts (pd.Series): Series of user prompts.
            model (str): OpenAI model name (e.g., "gpt-4o").
            system_prompt (str): Shared system message for chat context.
            logprobs (bool): Whether to request logprobs.
            top_logprobs (int): Number of top logprobs to include.
            temperature (float): Sampling temperature.
            output_file (str): Path to save intermediate JSONL file.
            url (str): API URL endpoint for the batch job.
            endpoint_id (str or None): Optional endpoint ID for Azure OpenAI or special routing.
        """
        # Step 1: Create the batch file
        make_batch(prompts, self.name, system_prompt, logprobs, top_logprobs, temperature, output_file, url)

        # Step 2: Upload the file
        try:
            upload = openai.files.create(file=open(output_file, "rb"), purpose="batch")
            file_id = upload.id
            print(f"Uploaded file ID: {file_id}")
        except OpenAIError as e:
            print(f"File upload failed: {e}")
            return None

        # Step 3: Submit the batch
        try:
            batch = openai.batches.create(
                input_file_id=file_id,
                endpoint=url,
                completion_window="24h",  # or "1h" for faster window if available
                endpoint_id=endpoint_id  # Optional, pass None for default
            )
            print(f"Batch submitted. ID: {batch.id}, Status: {batch.status}")
            return batch
        except OpenAIError as e:
            print(f"Batch submission failed: {e}")
            return None


class ClaudeModels(BatchModels):
    def __init__(self, api_key=None):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def make_batch(self,
                   prompts: pd.Series,
                   model: str,
                   system_prompt: str,
                   temperature=0,
                   output_file="claude_batch.jsonl",
                   url="/v1/messages"):
        """
        Creates a JSONL file for Claude batch processing via the Anthropic API.

        Args:
            prompts (pd.Series): A pandas Series of user prompts.
            model (str): Claude model name (e.g., "claude-3-opus-20240229").
            system_prompt (str): System instruction for Claude.
            temperature (float): Sampling temperature.
            output_file (str): File path to save the batch JSONL.
            url (str): API endpoint for Claude batch requests (default: "/v1/messages").
        """
        requests = []

        for idx, user_prompt in prompts.items():
            req = {
                "custom_id": str(idx),
                "method": "POST",
                "url": url,
                "body": {
                    "model": model,
                    "system": system_prompt,
                    "temperature": temperature,
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": user_prompt}
                    ]
                }
            }
            requests.append(req)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in requests:
                f.write(json.dumps(item) + "\n")

        print(f"Saved {len(requests)} prompts to {output_file}")

    def run_batch(self,
                  prompts: pd.Series,
                  model: str,
                  system_prompt: str,
                  temperature=0,
                  output_file="claude_batch.jsonl",
                  url="/v1/messages",
                  completion_window="24h"):
        """
        Submits a batch job to Claude API.

        Args:
            prompts (pd.Series): User prompts.
            model (str): Claude model name.
            system_prompt (str): System instruction.
            temperature (float): Sampling temperature.
            output_file (str): Output file for the batch JSONL.
            url (str): Claude endpoint URL.
            completion_window (str): Completion window, either "1h" or "24h".
        """
        self.make_batch(prompts, model, system_prompt, temperature, output_file, url)

        try:
            upload = self.client.files.create(file=open(output_file, "rb"), purpose="batch")
            file_id = upload.id
            print(f"Uploaded file with ID: {file_id}")
        except APIError as e:
            print(f"Upload failed: {e}")
            return None

        try:
            batch = self.client.batches.create(
                input_file_id=file_id,
                endpoint=url,
                completion_window=completion_window
            )
            print(f"Batch submitted. ID: {batch.id}, Status: {batch.status}")
            return batch
        except APIError as e:
            print(f"Batch creation failed: {e}")
            return None


class GeminiModels(BatchModels):
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the Gemini client.
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)

    def run_batch(self,
                  prompts: pd.Series,
                  model: str = "gemini-1.5-pro-latest",
                  system_prompt: str = "",
                  temperature: float = 0.0,
                  safety_settings: Optional[list] = None,
                  generation_config: Optional[dict] = None):
        """
        Submits prompts to Gemini one-by-one and returns responses in a dictionary.

        Args:
            prompts (pd.Series): Series of prompt strings.
            model (str): Gemini model name.
            system_prompt (str): Optional system-level prompt.
            temperature (float): Sampling temperature.
            safety_settings (list): Optional list of safety settings.
            generation_config (dict): Optional dict for config (e.g., max_tokens).
        
        Returns:
            dict: Mapping from prompt index to response text or error.
        """
        gemini_model = genai.GenerativeModel(
            model,
            system_instruction=system_prompt if system_prompt else None,
            safety_settings=safety_settings,
            generation_config=generation_config or {"temperature": temperature}
        )

        responses = {}

        for idx, prompt in prompts.items():
            try:
                response = gemini_model.generate_content(prompt)
                responses[idx] = response.text
                print(f"✅ Completed prompt {idx}")
            except Exception as e:
                print(f"❌ Error on prompt {idx}: {e}")
                responses[idx] = None

        return responses
    

# Register models for easy access