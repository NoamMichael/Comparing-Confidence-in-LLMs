## Version 1.0.0
# batch_processing.py
# This module provides classes and functions for batch processing of prompts using various LLM APIs.
# It supports OpenAI's GPT, Anthropic's Claude, and Google's Gemini models.
# It includes functionality for creating batch files, uploading them, and submitting batch jobs.
#--------------------------------------------------------------
# Next Steps:
#--------------------------------------------------------------

# 1. Add in function prompting method for each dataset.
# 2. Implement error handling and logging for API requests.



import json
import pandas as pd
import openai
import os
import anthropic
import google.generativeai as genai
from openai import OpenAIError
from abc import ABC, abstractmethod
from anthropic import Anthropic, AsyncAnthropic, APIError

from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                 MODEL CLASSES
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class BatchModels(ABC):
    def __init__(self, name: str, api_key_name: str):
        self.name = name
        self.api_key_name = api_key_name
        self.api_key = os.getenv(api_key_name)
        if not self.api_key:
            print(f"Warning: API key '{api_key_name}' not found in environment.")

    @abstractmethod
    def run_batch(self, prompts: pd.Series, **kwargs):
        pass

    @abstractmethod
    def make_batch(self, prompts: pd.Series, **kwargs):
        pass

class GPTModels(BatchModels):
    def __init__(self, name: str, api_key_name: str):
        super().__init__(name, api_key_name)
        if self.api_key:
            openai.api_key = self.api_key
        else:
            print("OpenAI API key not configured. GPT models will not work.")
    def make_batch(self, prompts: pd.Series, model: str, system_prompt: str, logprobs=True, top_logprobs=5,
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
        self.make_batch(prompts, self.name, system_prompt, logprobs, top_logprobs, temperature, output_file, url)

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
    def __init__(self, name: str, api_key_name: str):
        super().__init__(name, api_key_name)
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
            print("Anthropic API key not configured. Claude models will not work.")

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
    def __init__(self, name: str, api_key_name: str):
        """
        Initializes the Gemini client.
        """
        super().__init__(name, api_key_name)
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            print("Google API key not configured. Gemini models will not work.")

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

    def make_batch(self, prompts: pd.Series, **kwargs):
        """
        Gemini API does not use a batch file in this implementation.
        This method is a placeholder to satisfy the abstract class requirements.
        """
        print(f"Note: Gemini model '{self.name}' processes prompts sequentially and does not use batch files.")
        pass

    

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               MODEL REGISTRATION
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


models = {
    'GPT': {
        'class': GPTModels,
        'api_key_name': 'OPENAI_API_KEY',  # Environment variable for the API key
        'models': [
            'gpt-4',
            'gpt-3.5-turbo'
        ]
    },
    'Claude': {
        'class': ClaudeModels,
        'api_key_name': 'ANTHROPIC_API_KEY',  # Environment variable for the API key
        'models': [
            'claude-3-7-sonnet-20250219',
            'claude-3-haiku-20240307'
        ]
    },
    'Gemini': {
        'class': GeminiModels,
        'api_key_name': 'GOOGLE_API_KEY',  # Environment variable for the API key
        'models': [
            'gemini-1.5-flash',
            'gemini-2.5-pro-preview-06-05'
        ]
    }
}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                Make prompts for Qset
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def boolq_valid_prompts(df): 
    system_prompt = '''
    Analyze the following question and determine whether it is true or false. Your reasoning should not exceed 100 words. After your explanation, clearly state your answer.

    Question: ${Question}

    Response: {
    'Reasoning': <Your concise reasoning here. Max 100 words>
    'Answer': <True or False>
    'Confidence': <Probability that your answer is correct. As a float from 0.0 to 1.0>
    }

    All options have a non-zero probability of being correct. No option should have a probability of 0 or 1.
    Be modest about your certainty.  Do not provide any additional reasoning.
    '''



    header = ['Question ID', 'Full Prompt', 'System Prompt']
    output_rows = []
    for i, row in df.iterrows():
        full_prompt = system_prompt.replace('${Question}', str(row['Question'])) + 'Response: '
        output_rows.append([row['Question ID'], full_prompt, system_prompt])
    output_df = pd.DataFrame(output_rows, columns=header)
    return output_df

def halu_eval_prompts(): pass

def life_eval_prompts(df): 
    system_prompt = """
    You are a helpful assistant. Given the information provided about the person, first reason using what you know about population statistics and then make your best guess on how old they will live to. 
    Provide the probability that their true age at death falls within the range provided. Be modest about your certainty. The conversation is over once you answer the problem.

    Provide your response in a JSON format. For example:

    Response:
    {
        'Reasoning': '<Your concise reasoning>',
        'Answer': '<An integer value representing the age you believe the person will die at>',
        'Probability': '<A float between 0 and 1 representing the probability that the true age at death falls in the range>'
    }
    """
    header = ['Question ID', 'Full Prompt', 'System Prompt']

    output_rows = []
    for i, row in df.iterrows():
        full_prompt = row['Question Prompt'] + ' '+ row['Confidence Prompt'] + '\nResponse:\n'
        output_rows.append([row['Question ID'], full_prompt, system_prompt])
    output_df = pd.DataFrame(output_rows, columns=header)
    return

def lsat_ar_test_prompts(df):  ## Takes in the csv from GitHub and outputs prompts
  ## Takes in a dataframe in the form:
  ## | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  ## |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |
  ##
  ## Returns a dataframe in the form:
  ## | Question ID | Full Prompt 1 | Full Prompt 2 |
  ## |     (Int)       |    (Str)      |    (Str)      |
  sys_prompt1 = '''
    Given the following question, analyze the options, and provide a concise reasoning for your selected answer. Your reasoning should not exceed 100 words. After your explanation, clearly state your answer by choosing one of the options listed (A, B, C, D, or E).

    Question: ${Question}
    Options:
    A) ${Option A}
    B) ${Option B}
    C) ${Option C}
    D) ${Option D}
    E) ${Option E}

    Please provide your reasoning first, limited to 100 words, and consider how certain you should be of your answer.
    Reasoning: <Your concise reasoning here. Max 100 words>
    '''
  sys_prompt2 = '''
    Based on the reasoning above, Provide the best answer and the likelihood that each option is correct from 0.0 to 1.0 in a JSON format. The probabilities should sum to 1.0. For example:

    {
    'A': <Probability choice A is correct. As a float from 0.0 to 1.0>,
    'B': <Probability choice B is correct. As a float from 0.0 to 1.0>,
    'C': <Probability choice C is correct. As a float from 0.0 to 1.0>,
    'D': <Probability choice D is correct. As a float from 0.0 to 1.0>,
    'E': <Probability choice E is correct. As a float from 0.0 to 1.0>,
    'Answer': <Your answer choice here, as a single letter and nothing else.>
    }

    All options have a non-zero probability of being correct. No option should have a probability of 0 or 1.
    Be modest about your certainty.  Do not provide any additional reasoning.

    Response:
    '''
  columns = df.columns
  num_options = columns.str.contains('Option').astype(int).sum()

  #----------------------------------------------------------------------------#
  ## Check if DF is formatted properly
  error_text = f'''Make sure dataframe is in following format:
  | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |

  The current format of Dataframe is: {columns}
  '''
  ['Question ID', 'Question', 'Correct Answer Letter']
  if num_options < 2:
    raise Exception(error_text)

  #----------------------------------------------------------------------------#
  ## Initialize Output dataframe:
  header = ['Question ID', 'Full Prompt', 'Response Prompt', 'System Prompt']
  output_df = pd.DataFrame(columns = header)

  #----------------------------------------------------------------------------#

  ## Format questions for benchmark
  letters = ['A', 'B', 'C', 'D', 'E']
  options = ['Option A', 'Option B', 'Option C', 'Option D', 'Option E']

  for i in range(len(df)):
    question = df['Question'][i]

    sys_prompt_temp1 = sys_prompt1
    sys_prompt_temp2 = sys_prompt2
    ## Reformat system prompt in order to fit number of options in benchmark
    if type(df['Option E'][i]) == float: ## ABCD
      sys_prompt_temp1 = (sys_prompt1
                    .replace('(A, B, C, D, or E)', '(A, B, C, or D)')
                    .replace('E) ${Option E}', '')
          )
      sys_prompt_temp2 = (sys_prompt2
                    .replace('(A, B, C, D, or E)', '(A, B, C, or D)')
                    .replace('E) ${Option E}', '')
          )
      if type(df['Option D'][i]) == float: ## ABC
        sys_prompt_temp1 = (sys_prompt_temp1
                      .replace('(A, B, C, or D)', '(A, B, or C)')
                      .replace('D) ${Option D}', '')
            )
        sys_prompt_temp2 = (sys_prompt_temp2
                    .replace('(A, B, C, or D)', '(A, B, or C)')
                    .replace('D) ${Option D}', '')
          )

        if type(df['Option C'][i]) == float: ## AB
          sys_prompt_temp1 = (sys_prompt_temp1
                        .replace('(A, B, or C)', '(A or B)')
                        .replace('C) ${Option C}', '')
              )
          sys_prompt_temp2 = (sys_prompt_temp2
                      .replace('(A, B, or C)', '(A or B)')
                      .replace('C) ${Option C}', '')
            )

    option_text = df[options[:num_options]].iloc[i].to_list()
    ## Prompt for specific question
    new_prompt = sys_prompt_temp1.replace('${Question}', question)
    for j in range(num_options): ## This for loop allows for dynamic question amounts
        new_prompt = new_prompt.replace(f'${{Option {letters[j]}}}', str(option_text[j]))


    ## Add formatted prompts.
    ## Note that this is formatted to llama so changes may be needed down the line.
    prompts1 = (new_prompt.split('<Your concise reasoning here. Max 100 words>')[0]) ## Specific prompt for question

    prompts2 = (sys_prompt_temp2) ## Generic prompt for question confidence
    output_df.loc[i] = [df['Question ID'].iloc[i], prompts1, prompts2, sys_prompt1]

  return output_df

def math_500_prompts(df):
    system_prompt = """
    You are a helpful math assistant. When prompted with problem, provide your step by step reasoning and final answer. Your answer can be a number or a mathematical expression.
    After providing your reasoning and answer, give a probability that you are correct. Be modest about your certainty. The conversation is over once you answer the problem. Your answer should be in a JSON format. 
    For example:

    Problem: <question>

    Response:
    {
        'Reasoning': '<Your step by step reasoning>',
        'Answer': '<Number or simplified mathematical expression>'
        'Confidence': '<Probability that your answer is correct. As a float from 0.0 to 1.0>'
    }
    """

    prefix = 'Answer the problem below. Provide your step by step reasoning, answer, and the probability that your answer is correct. Be modest about your certainty. The conversation is over once you answer the problem.'

    question = ''

    suffix = '\nResponse:\n'

    header = ['Question ID', 'Full Prompt', 'System Prompt']

    output_rows = []
    for i, row in df.iterrows():
        full_prompt = prefix + row['Question'] + suffix
        output_rows.append([row['Question ID'], full_prompt, system_prompt])
    output_df = pd.DataFrame(output_rows, columns=header)
    return output_df

def sat_en_prompts(df): ## Maybe edit the prompts from Qset
    system_prompt = """
    You are a helpful assistant taking the SAT. Your role is to read the passage and answer the question about the passage. First reason through the options using context and quotes from the text. Then, provide your answer and the probability that each answer is correct.
    All options have a non-zero probability of being correct. No option should have a probability of 0 or 1. Be modest about your certainty.  Do not provide any additional reasoning. Your answer should be in a JSON format.
    For example:

    Passage: <Passage>

    Q: <Question about the passage>

    Answer Choices: <Answer choices A-D>

    Response:
    {
        'Reasoning': '<YOUR reasoning>',
        'Answer': '<YOUR final answer>',
        'Confidence': '<Probability that YOUR answer is correct. As a float from 0.0 to 1.0>'
    }
    """
    prefix = 'Directions: Read the passage below and consider which letter is the best answer to the question'

    suffix = '\nResponse:\n'

    header = ['Question ID', 'Full Prompt', 'System Prompt']

    output_rows = []
    for i, row in df.iterrows():
        passage_question_options = row['query'].split('A: Among A through D, the answer is')[0]
        full_prompt = prefix + passage_question_options + suffix

        output_rows.append([row['Question ID'], full_prompt, system_prompt])
    output_df = pd.DataFrame(output_rows, columns=header)
    return output_df

def sciq_test_prompts(df): 
  sys_prompt1 = '''
        Given the following question, analyze the options, and provide a concise reasoning for your selected answer. Your reasoning should not exceed 100 words. After your explanation, clearly state your answer by choosing one of the options listed (A, B, C, or D).

        Question: ${Question}
        Options:
        A) ${Option A}
        B) ${Option B}
        C) ${Option C}
        D) ${Option D}


        Please provide your reasoning first, limited to 100 words, and consider how certain you should be of your answer.
        Reasoning: <Your concise reasoning here. Max 100 words>
        '''
  sys_prompt2 = '''
        Based on the reasoning above, Provide the best answer and the likelihood that each option is correct from 0.0 to 1.0 in a JSON format. The probabilities should sum to 1.0. For example:

        {
        'A': <Probability choice A is correct. As a float from 0.0 to 1.0>,
        'B': <Probability choice B is correct. As a float from 0.0 to 1.0>,
        'C': <Probability choice C is correct. As a float from 0.0 to 1.0>,
        'D': <Probability choice D is correct. As a float from 0.0 to 1.0>,
        'Answer': <Your answer choice here, as a single letter and nothing else.>
        }

        All options have a non-zero probability of being correct. No option should have a probability of 0 or 1.
        Be modest about your certainty.  Do not provide any additional reasoning.

        Response:
        '''

  columns = df.columns
  num_options = columns.str.contains('Option').astype(int).sum()

  #----------------------------------------------------------------------------#
  ## Check if DF is formatted properly
  error_text = f'''Make sure dataframe is in following format:
  | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |

  The current format of Dataframe is: {columns}
  '''
  ['Question ID', 'Question', 'Correct Answer Letter']
  if num_options < 2:
    raise Exception(error_text)

  #----------------------------------------------------------------------------#
  ## Initialize Output dataframe:
  header = ['Question ID', 'Full Prompt', 'Response Prompt', 'System Prompt', 'Response']
  output_df = pd.DataFrame(columns = header)

  #----------------------------------------------------------------------------#

  ## Format questions for benchmark
  letters = ['A', 'B', 'C', 'D', 'E']
  options = ['Option A', 'Option B', 'Option C', 'Option D']

  for i in range(len(df)):
    question = df['Question'][i]

    sys_prompt_temp1 = sys_prompt1
    sys_prompt_temp2 = sys_prompt2
    ## Reformat system prompt in order to fit number of options in benchmark
    if type(df['Option E'][i]) == float: ## ABCD
      sys_prompt_temp1 = (sys_prompt1
                    .replace('(A, B, C, D, or E)', '(A, B, C, or D)')
                    .replace('E) ${Option E}', '')
          )
      sys_prompt_temp2 = (sys_prompt2
                    .replace('(A, B, C, D, or E)', '(A, B, C, or D)')
                    .replace('E) ${Option E}', '')
          )
      if type(df['Option D'][i]) == float: ## ABC
        sys_prompt_temp1 = (sys_prompt_temp1
                      .replace('(A, B, C, or D)', '(A, B, or C)')
                      .replace('D) ${Option D}', '')
            )
        sys_prompt_temp2 = (sys_prompt_temp2
                    .replace('(A, B, C, or D)', '(A, B, or C)')
                    .replace('D) ${Option D}', '')
          )

        if type(df['Option C'][i]) == float: ## AB
          sys_prompt_temp1 = (sys_prompt_temp1
                        .replace('(A, B, or C)', '(A or B)')
                        .replace('C) ${Option C}', '')
              )
          sys_prompt_temp2 = (sys_prompt_temp2
                      .replace('(A, B, or C)', '(A or B)')
                      .replace('C) ${Option C}', '')
            )

    option_text = df[options[:num_options]].iloc[i].to_list()
    ## Prompt for specific question
    new_prompt = sys_prompt_temp1.replace('${Question}', question)
    for j in range(num_options): ## This for loop allows for dynamic question amounts
        new_prompt = new_prompt.replace(f'${{Option {letters[j]}}}', str(option_text[j]))


    ## Add formatted prompts.
    ## Note that this is formatted to llama so changes may be needed down the line.
    prompts1 = (new_prompt.split('<Your concise reasoning here. Max 100 words>')[0]) ## Specific prompt for question

    prompts2 = (sys_prompt_temp2) ## Generic prompt for question confidence
    output_df.loc[i] = [df['Question ID'].iloc[i], prompts1, prompts2, sys_prompt1, sys_prompt2]

  return output_df
    
def truthful_qa_prompts(): pass

## Map functions to dataset

functions_map = {
    'boolq_valid': boolq_valid_prompts,
    'halu_eval': halu_eval_prompts,
    'life_eval': life_eval_prompts,
    'lsat_ar_test': lsat_ar_test_prompts,
    'math_500': math_500_prompts,
    'sat_en': sat_en_prompts,
    'sciq_test': sciq_test_prompts,
    'truthful_qa': truthful_qa_prompts
}



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                HELPER FUNCTIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def init_models(models_dict):
    """
    Initializes model instances based on a configuration dictionary.

    Args:
        models_dict (dict): A dictionary containing the configuration for model groups.
                            See the global 'models' variable for an example structure.

    Returns:
        list: A list of initialized model instances.
    """
    print('Initializing Models:')
    initialized_models = []
    for model_group, config in models_dict.items():
        print(f'{model_group}:')
        model_class = config['class']
        api_key_name = config['api_key_name']
        print(f'  Key Name: {api_key_name}')

        for model_name in config['models']:
            # The __init__ of the model class handles getting the API key
            model_instance = model_class(name=model_name, api_key_name=api_key_name)
            initialized_models.append(model_instance)
            print(f'    Initialized: {model_name}')

    print(f'\nTotal Models Initialized: {len(initialized_models)}')
    print('-' * 42)
    return initialized_models

def import_datasets(folder_path="Formatted Benchmarks/"):
    """
    Imports all datasets from a specified folder that end with '_formatted.csv'.

    Args:
        folder_path (str): The path to the folder containing the datasets.

    Returns:
        dict: A dictionary where keys are the dataset names (without the suffix)
              and values are the corresponding pandas DataFrames.
    """
    datasets = {}
    print(f"Importing datasets from '{folder_path}'...")
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith("_formatted.csv"):
                dataset_name = filename.replace("_formatted.csv", "")
                file_path = os.path.join(folder_path, filename)
                try:
                    datasets[dataset_name] = pd.read_csv(file_path)
                    print(f"  Successfully imported: {dataset_name}")
                except Exception as e:
                    print(f"  Error importing {filename}: {e}")
            else:
                print(f"  Skipping non-formatted file: {filename}")
    except FileNotFoundError:
        print(f"Error: The directory '{folder_path}' was not found.")
    
    print(f"\nTotal Datasets Imported: {len(datasets)}")
    print('-' * 42)
    return datasets


def format_prompts(question_set_dict, metadata_path='prompts_metadata.json'):
    """
    Processes a list of DataFrames to format prompts using metadata.

    Args:
        df_list (list): A list of pandas DataFrames to be processed.
        metadata_path (str): The path to the JSON file containing prompt metadata.

    Returns:
        dict: A dictionary of processed DataFrames with formatted prompts.
    """
    with open(metadata_path, 'r') as f:
        prompts_metadata = json.load(f)

    for df_name, df in question_set_dict:
        
        format_function = functions_map[df_name]

        
        
        print(f"Processing DataFrame with metadata...")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               MAIN EXECUTION BLOCK
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Process:
# 1. Init models
# 2. Import all question sets
# 3. Make fully prompt dataframe [QID, prompt1, prompt2, system]
# 3. For each model, for each dataset write and run a batch
#       - Make a skip dataset dict (e.g. skip LSAT, SCIQ, BoolQ for Claude)
# 4. 

if __name__ == '__main__':
    # Example of how to use the init_models function
    # Make sure your .env file has OPENAI_API_KEY, ANTHROPIC_API_KEY, and GOOGLE_API_KEY

    all_models = init_models(models)
    print("Returned model instances:", all_models)

    # Example of how to use the import_datasets function
    all_datasets = import_datasets()
    print("\nAvailable datasets:", list(all_datasets.keys()))

    formatted_prompts = format_prompts(all_datasets)