## Version 1.0.0
# batch_processing.py
# This module provides classes and functions for batch processing of prompts using various LLM APIs.
# It supports OpenAI's GPT, Anthropic's Claude, and Google's Gemini models.
# It includes functionality for creating batch files, uploading them, and submitting batch jobs.
#--------------------------------------------------------------
# Next Steps:
#--------------------------------------------------------------

# Occasional errors on Gem/ Claude errors. Google cloud yells at me sometimes about formatting. Claude can't get the response right.



import json
import pandas as pd
import openai
import os
import time
import anthropic
#import google.generativeai as genai
import google.genai as genai
from google.genai import types

from openai import OpenAIError
from abc import ABC, abstractmethod
from anthropic import Anthropic, AsyncAnthropic, APIError

from typing import Optional
from dotenv import load_dotenv
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

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
    def run_batch(self, prompts: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def make_batch(self, prompts: pd.DataFrame, **kwargs):
        pass


class GPTModels(BatchModels):
    def __init__(self, name: str, api_key_name: str):
        super().__init__(name, api_key_name)
        self.file_map = {}
        if self.api_key:
            openai.api_key = self.api_key
        else:
            print("OpenAI API key not configured. GPT models will not work.")
    def make_batch(self, prompts: pd.DataFrame, model: str, system_prompt: str, dataset_name: str, logprobs=True, top_logprobs=5,
                  temperature=0, url="/v1/chat/completions"):
        """
        Creates a JSONL file for the OpenAI batch API from a series of prompts using the chat completions format.

        Args:
            prompts (pd.DataFrame): A pandas DataFrame with 'Question ID' and 'Full Prompt' columns.
            model (str): The name of the OpenAI model to use (e.g., "gpt-4").
            system_prompt (str): The shared system message for all prompts.
            dataset_name (str): The name of the dataset, used for the output filename.
            logprobs (bool): Whether to request logprobs in the output.
            top_logprobs (int): Number of top logprobs to include.
            temperature (float): Sampling temperature for generation.
            url (str): API endpoint URL, defaulting to the chat completions endpoint.
        """
        output_dir = f"Batches/{self.name}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{dataset_name}_batch.jsonl")
        
        requests = []

        for _, row in prompts.iterrows():
            if model == 'o3-2025-04-16':
                logprobs = False
                request = {
                    "custom_id": str(row['Question ID']),
                    "method": "POST",
                    "url": url,
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": row['Full Prompt']}
                        ],
                        "temperature": 1,
                        "max_completion_tokens": 2048  # adjust as needed
                    }
                }
            else:    
                request = {
                    "custom_id": str(row['Question ID']),
                    "method": "POST",
                    "url": url,
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": row['Full Prompt']}
                        ],
                        "temperature": temperature,
                        #"max_tokens": 2048  # adjust as needed
                    }
                }

                # Only include logprobs fields if requested

                if logprobs:
                    request["body"]["logprobs"] = True
                    request["body"]["top_logprobs"] = 5


            requests.append(request)

        # Write each JSON object as a line in the JSONL file
        with open(output_file, "w", encoding="utf-8") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        print(f"Saved {len(requests)} prompts to {output_file}")
        return output_file

    def run_batch(self,
                  prompts: pd.DataFrame,
                  model: str,
                  system_prompt: str,
                  dataset_name: str,
                  logprobs=True,
                  top_logprobs=5,
                  temperature=0,
                  url="/v1/chat/completions",
                  endpoint_id=None):
        """
        Creates and submits a batch job to the OpenAI API using a series of prompts.

        Args:
            prompts (pd.DataFrame): DataFrame with 'Question ID' and 'Full Prompt' columns.
            model (str): OpenAI model name (e.g., "gpt-4o").
            system_prompt (str): Shared system message for chat context.
            dataset_name (str): The name of the dataset, used for the output filename.
            logprobs (bool): Whether to request logprobs.
            top_logprobs (int): Number of top logprobs to include.
            temperature (float): Sampling temperature.
            url (str): API URL endpoint for the batch job.
            endpoint_id (str or None): Optional endpoint ID for Azure OpenAI or special routing.
        """
        ## First read the results_metadata.json file

        with open("results_metadata.json", "r") as f:
            metadata = json.load(f)




        # Step 1: Create the batch file
        output_file = self.make_batch(prompts, model, system_prompt, dataset_name, logprobs, top_logprobs, temperature, url)

        # Step 2: Upload the file
        try:
            with open(output_file, "rb") as f:
                upload = openai.files.create(
                    file=f, 
                    purpose="batch",
                    
                    )
            file_id = upload.id
            print(f"Uploaded file ID: {file_id}")
            self.file_map[dataset_name] = file_id
        except OpenAIError as e:
            print(f"File upload failed for model {model}: {e}")
            return None

        # Step 3: Submit the batch
        try:
            batch = openai.batches.create(
                input_file_id=file_id,
                endpoint=url,
                completion_window="24h",  # or "1h" for faster window if available
            )
            print(f"Batch submitted. ID: {batch.id}, Status: {batch.status}")
            metadata.setdefault(dataset_name, {})
            metadata[dataset_name].setdefault("models", {})
            metadata[dataset_name]["models"].setdefault(self.name, {})

            metadata[dataset_name]["models"][self.name]['file_id'] = file_id
            metadata[dataset_name]["models"][self.name]['batch_id'] = batch.id
            # Save the modified dict back to the same file
            with open("results_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            return batch
        except OpenAIError as e:
            print(f"Batch submission failed for model {model}: {e}")

                # 2. Modify the dictionary (example: add a new key)

            return None

class ClaudeModels(BatchModels):
    def __init__(self, name: str, api_key_name: str):
        super().__init__(name, api_key_name)
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
            print("Anthropic API key not configured. Claude models will not work.")

    def make_batch(self, prompts: pd.DataFrame, **kwargs):
        """
        This method is not used for the Claude batch API implementation.
        """
        print("Note: ClaudeModels does not use a separate make_batch function.")
        pass

    def run_batch(self,
                  prompts: pd.DataFrame,
                  model: str,
                  system_prompt: str,
                  dataset_name: str,
                  temperature=0,
                  **kwargs):
        """
        Processes prompts using the Anthropic Messages Batch API.

        Args:
            prompts (pd.DataFrame): DataFrame with 'Question ID' and 'Full Prompt' columns.
            model (str): Claude model name.
            system_prompt (str): System instruction.
            dataset_name (str): The name of the dataset.
            temperature (float): Sampling temperature.
        """
        if not self.client:
            print("Claude client not initialized. Skipping batch.")
            return
        
        # 1. Load JSON file into dict
        with open("results_metadata.json", "r") as f:
            metadata = json.load(f)

        requests = []
        for _, row in prompts.iterrows():
            requests.append(
                Request(
                    custom_id=str(row['Question ID']),
                    params=MessageCreateParamsNonStreaming(
                        model=model,
                        max_tokens=512,
                        messages=[{
                            "role": "user",
                            "content": row['Full Prompt']
                        }],
                        system=system_prompt,
                        temperature=temperature
                    )
                )
            )
        
        try:
            batch_job = self.client.messages.batches.create(requests=requests)
            print(f"Successfully submitted batch job for {model} on {dataset_name}. Batch ID: {batch_job.id}")

            metadata.setdefault(dataset_name, {})
            metadata[dataset_name].setdefault("models", {})
            metadata[dataset_name]["models"].setdefault(self.name, {})

            metadata[dataset_name]["models"][self.name]['file_id'] = dataset_name
            metadata[dataset_name]["models"][self.name]['batch_id'] = batch_job.id
            # Save the modified dict back to the same file
            with open("results_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        except APIError as e:
            print(f"Error submitting batch job for {model} on {dataset_name}: {e}")

class GeminiModels(BatchModels):
    def __init__(self, name: str, api_key_name: str):
        """
        Initializes the Gemini client.
        """
        super().__init__(name, api_key_name)
        if self.api_key:
            #genai.configure(api_key=self.api_key)
            self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})
        else:
            print("Google API key not configured. Gemini models will not work.")

    def run_batch(self,
                  prompts: pd.DataFrame,
                  model: str,
                  dataset_name: str,
                  system_prompt: str = "",
                  temperature: float = 0.0,
                  **kwargs):
        """
        Gemini API does not support asynchronous batch processing in the same way as OpenAI or Claude.
        This function will create a batch file but will not submit it.
        """
        self.make_batch(prompts, model=model, dataset_name=dataset_name, system_prompt=system_prompt, temperature=temperature, **kwargs)
        print(f"Note: Gemini model '{self.name}' batch file created. Manual processing required.")
        time.sleep(45) #Rate limit
        pass

    def make_batch(self,
                prompts: pd.DataFrame,
                model: str,
                dataset_name: str,
                system_prompt: str = "",
                temperature: float = 0.0,
                **kwargs):
        """
        Creates a JSONL file for Gemini batch predictions using Vertex AI format.

        Args:
            prompts (pd.DataFrame): A DataFrame with 'Question ID' and 'Full Prompt' columns.
            model (str): Model name (kept for compatibility).
            dataset_name (str): Used to name the output file.
            system_prompt (str): Optional system-level instruction.
            temperature (float): Sampling temperature (optional).
        """
        output_dir = f"Batches/{self.name}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{dataset_name}_batch.json")

        requests = []
        for _, row in prompts.iterrows():
            item = {
                "key": str(row["Question ID"]),
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": system_prompt}]
                        },
                        {
                            "role": "user",
                            "parts": [{"text": row["Full Prompt"]}]
                        }
                    ]

                }
            }
            requests.append(item)
        

        with open(output_file, "w", encoding="utf-8") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")


        print(f"Saved {len(requests)} prompts to {output_file} for Gemini/Vertex AI Batch Prediction.")


        uploaded_batch_requests = self.client.files.upload(
            file=output_file,
            config=types.UploadFileConfig(display_name='batch-input-file')
        )
        print(f"Uploaded file: {uploaded_batch_requests.name}")
        batch_job_from_file = self.client.batches.create(
            model=self.name,
            src=uploaded_batch_requests.name,
            config={
                'display_name': f'{self.name}_{dataset_name}',
            }
        )
        print(f"Created batch job from file: {batch_job_from_file.name}")

        job_name = batch_job_from_file.name

        with open("results_metadata.json", "r") as f:
            metadata = json.load(f)

            metadata.setdefault(dataset_name, {})
            metadata[dataset_name].setdefault("models", {})
            metadata[dataset_name]["models"].setdefault(self.name, {})

            metadata[dataset_name]["models"][self.name]['file_id'] = 'none'
            metadata[dataset_name]["models"][self.name]['batch_id'] = job_name
            # Save the modified dict back to the same file
            with open("results_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        return output_file
        

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               MODEL REGISTRATION
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                Make prompts for Qset
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def boolq_valid_prompts(df): 
    system_prompt = '''
You are a helpful assistant. Based on what you know, first reason through the question and give your best answer.
Additionally, provide your confidence in the answer as a float from 0.0 to 1.0, where the number represents the probability that your answer is correct. 
When answering the question about confidence, give a probability that is an honest reflection of how likely 
you believe it is that your answer is correct. Your response should be in a JSON format. For example:

Question: <Question>

Response: {
    "Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
    "Answer": "<True or False>",
    "Confidence": "<Probability that your answer is correct. As a float from 0.0 to 1.0>"
}

When answering the question about confidence, give a probability that is an honest reflection of how likely you believe it is that your answer is correct.
'''
    strip_portion = '''Response: {
    "Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
    "Answer": "<True or False>",
    "Confidence": "<Probability that your answer is correct. As a float from 0.0 to 1.0>"
}

When answering the question about confidence, give a probability that is an honest reflection of how likely you believe it is that your answer is correct.
'''

    header = ['Question ID', 'Full Prompt', 'System Prompt']
    output_rows = []
    for i, row in df.iterrows():
        full_prompt = 'Directions:\n' + system_prompt + 'Question: '+ str(row['Question']).capitalize() + '?\n' + 'Response: '
        output_rows.append([row['Question ID'], full_prompt, system_prompt])
    output_df = pd.DataFrame(output_rows, columns=header)
    return output_df

def halu_eval_qa_prompts(df):
    '''
    1. Provide context
    2. Provide Question
    3. Provide real/fake answer rename QID with suffix _r / _h
    4. 

    '''
    output_df = pd.DataFrame(columns=['Question ID', 'Full Prompt', 'System Prompt'])
    q_types = ['_r', '_h']
    system_prompt = '''You are a helpful assistant. Based on the context provided, answer the question as accurately as possible. 
Additionally, provide your confidence in the answer as a float from 0.0 to 1.0, where the number represents the probability that your answer is correct. 
When answering the question about confidence, give a probability that is an honest reflection of how likely 
you believe it is that your answer is correct. Your response should be in a JSON format.'''

    example = ''' For example:
Context: <Context>
Question: <Question>
Response:
\u007b
    "Answer": "<Your answer>",
    "Confidence": "<The probability that your answer is correct as a float from 0.0 to 1.0>"
\u007d'''
    full_system = system_prompt + example
    for i, row in df.iterrows():
        knowledge = row['knowledge']
        question = row['Question']
        for type in q_types:
            new_row = pd.DataFrame(columns=['Question ID', 'Full Prompt', 'System Prompt'])
            if type == '_r':
                response = row['right_answer']
            else:
                response = row['hallucinated_answer']
            qid = str(row['Question ID']) + type

            full_prompt = f'''Directions: {full_system}\n

Context: {knowledge}\n
Question: {question}\n
Response:
\u007b
    "Answer": "{response}",
    "Confidence": "'''
            
            new_row['Question ID'] = pd.Series([qid])
            new_row['Full Prompt'] = pd.Series([full_prompt])
            new_row['System Prompt'] = pd.Series([system_prompt + example])

            output_df = pd.concat([output_df, new_row])
    print(f'HALUEVAL LENGTH: {len(output_df)}')
    return output_df
        
def life_eval_prompts(df): 
    system_prompt = """You are a helpful assistant. Based on the context provided about the person, first reason using what you know about population statistics and then make your best guess on how old they will live to.
Additionally, provide your confidence in the answer as a float from 0.0 to 1.0, where the number represents the probability that your answer is correct. 
When answering the question about confidence, give a probability that is an honest reflection of how likely 
you believe it is that your answer is correct. Your response should be in a JSON format. For example:

Question: <Question>
Response:
{
    "Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
    "Answer": "<An integer value representing the age you believe the person will die at>",
    "Confidence": "<A float between 0 and 1 representing the probability that the true age at death falls in the range>"
}"""
    header = ['Question ID', 'Full Prompt', 'System Prompt']

    output_rows = []
    for i, row in df.iterrows():
        full_prompt = 'Directions:\n' + system_prompt + '\nQuestion:\n' + row['Question Prompt'] + ' '+ row['Confidence Prompt'] + '\nResponse:\n'
        full_prompt = full_prompt.replace(
            'How certain that your answer is within 1 years of the true value?',
            'How certain are you that your answer is within 1 year of the true value?'
        )
        full_prompt = full_prompt.replace('How certain that your', 'How certain are you that your')
        output_rows.append([row['Question ID'], full_prompt, system_prompt])
    output_df = pd.DataFrame(output_rows, columns=header)
    return output_df

def lsat_ar_test_prompts(df):  ## Takes in the csv from GitHub and outputs prompts
  ## Takes in a dataframe in the form:
  ## | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  ## |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |
  ##
  ## Returns a dataframe in the form:
  ## | Question ID | Full Prompt 1 | Full Prompt 2 |
  ## |     (Int)       |    (Str)      |    (Str)      |
  sys_prompt1 = '''You are a helpful assistant. Given the following question, analyze the options, and provide a concise reasoning for your selected answer. Your reasoning should not exceed 100 words. 
Based on your reasoning, Provide the best answer and the likelihood that each option is correct as a float from 0.0 to 1.0 in a JSON format. The probabilities should sum to 1. For example:

Question: <Question>
Options:
A) <Option A>
B) <Option B>
C) <Option C>
D) <Option D>
E) <Option E>

Response:

{
"Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
"Answer": "<Your answer choice here, as a single letter and nothing else>",
"A": "<Probability choice A is correct. As a float from 0.0 to 1.0>",
"B": "<Probability choice B is correct. As a float from 0.0 to 1.0>",
"C": "<Probability choice C is correct. As a float from 0.0 to 1.0>",
"D": "<Probability choice D is correct. As a float from 0.0 to 1.0>",
"E": "<Probability choice E is correct. As a float from 0.0 to 1.0>"
}

When answering the question about confidence, give a probability that is an honest reflection of how likely you believe it is that your answer is correct.
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
  header = ['Question ID', 'Full Prompt', 'System Prompt']
  output_df = pd.DataFrame(columns = header)

  #----------------------------------------------------------------------------#

  ## Format questions for benchmark
  letters = ['A', 'B', 'C', 'D', 'E']
  options = ['Option A', 'Option B', 'Option C', 'Option D', 'Option E']

  for i in range(len(df)):
    question = df['Question'][i].split(' Answer Choices: ')[0]

    sys_prompt_temp1 = sys_prompt1

    ## Reformat system prompt in order to fit number of options in benchmark
    if type(df['Option E'][i]) == float: ## ABCD
      sys_prompt_temp1 = (sys_prompt1
                    .replace('(A, B, C, D, or E)', '(A, B, C, or D)')
                    .replace('E) ${Option E}', '')
          )

      if type(df['Option D'][i]) == float: ## ABC
        sys_prompt_temp1 = (sys_prompt_temp1
                      .replace('(A, B, C, or D)', '(A, B, or C)')
                      .replace('D) ${Option D}', '')
            )

        if type(df['Option C'][i]) == float: ## AB
          sys_prompt_temp1 = (sys_prompt_temp1
                        .replace('(A, B, or C)', '(A or B)')
                        .replace('C) ${Option C}', '')
              )


    option_text = df[options[:num_options]].iloc[i].to_list()
    ## Prompt for specific question
    new_prompt = sys_prompt_temp1.split('Response:')[0].replace('<Question>', question).split('For example:')[1].replace('Question:', 'Context:').replace('.Q: ', '.\nQuestion: ')
    for j in range(num_options): ## This for loop allows for dynamic question amounts
        new_prompt = new_prompt.replace(f'<Option {letters[j]}>', str(option_text[j]))


    ## Add formatted prompts.
    ## Note that this is formatted to llama so changes may be needed down the line.
    prompts1 = 'Directions:\n' + sys_prompt1 + (new_prompt.split('<Your concise reasoning here. Max 100 words>')[0])  + '\nResponse:\n'## Specific prompt for question
    prompts1 = prompts1.replace(
            'Question: <Question>',
            '''Context: <Context>
Question: <Question>
''')
    output_df.loc[i] = [df['Question ID'].iloc[i], prompts1,  sys_prompt1]

  return output_df

def sciq_test_prompts(df):  ## Takes in the csv from GitHub and outputs prompts
  ## Takes in a dataframe in the form:
  ## | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  ## |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |
  ##
  ## Returns a dataframe in the form:
  ## | Question ID | Full Prompt 1 | Full Prompt 2 |
  ## |     (Int)       |    (Str)      |    (Str)      |
  sys_prompt1 = '''You are a helpful assistant. Given the following question, analyze the options, and provide a concise reasoning for your selected answer. Your reasoning should not exceed 100 words. 
Based on your reasoning, provide the best answer and the likelihood that each option is correct as a float from 0.0 to 1.0 in a JSON format. The probabilities should sum to 1. For example:

Question: <Question>
Options:
A) <Option A>
B) <Option B>
C) <Option C>
D) <Option D>

Response:
{
"Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
"Answer": "<Your answer choice here, as a single letter and nothing else>",
"A": "<Probability choice A is correct. As a float from 0.0 to 1.0>",
"B": "<Probability choice B is correct. As a float from 0.0 to 1.0>",
"C": "<Probability choice C is correct. As a float from 0.0 to 1.0>",
"D": "<Probability choice D is correct. As a float from 0.0 to 1.0>"
}

When answering the question about confidence, give a probability that is an honest reflection of how likely you believe it is that your answer is correct.
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
  header = ['Question ID', 'Full Prompt', 'System Prompt']
  output_df = pd.DataFrame(columns = header)

  #----------------------------------------------------------------------------#

  ## Format questions for benchmark
  letters = ['A', 'B', 'C', 'D']
  options = ['Option A', 'Option B', 'Option C', 'Option D']

  for i in range(len(df)):
    question = df['Question'][i].split(' Answer Choices: ')[0]

    sys_prompt_temp1 = sys_prompt1

    ## Reformat system prompt in order to fit number of options in benchmark
    '''
    if type(df['Option E'][i]) == float: ## ABCD
      sys_prompt_temp1 = (sys_prompt1
                    .replace('(A, B, C, D, or E)', '(A, B, C, or D)')
                    .replace('E) ${Option E}', '')
          )

      if type(df['Option D'][i]) == float: ## ABC
        sys_prompt_temp1 = (sys_prompt_temp1
                      .replace('(A, B, C, or D)', '(A, B, or C)')
                      .replace('D) ${Option D}', '')
            )

        if type(df['Option C'][i]) == float: ## AB
          sys_prompt_temp1 = (sys_prompt_temp1
                        .replace('(A, B, or C)', '(A or B)')
                        .replace('C) ${Option C}', '')
              )
    '''

    option_text = df[options[:num_options]].iloc[i].to_list()
    ## Prompt for specific question
    new_prompt = sys_prompt_temp1.split('Response:')[0].replace('<Question>', question).split('For example:')[1]#.replace('Question:', 'Premise:') ## Uncomment for Qset with premise.
    for j in range(num_options): ## This for loop allows for dynamic question amounts
       new_prompt = new_prompt.replace(f'<Option {letters[j]}>', str(option_text[j]))

    ## Add formatted prompts.
    ## Note that this is formatted to llama so changes may be needed down the line.
    prompts1 = 'Directions:\n' + sys_prompt1 + (new_prompt.split('<Your concise reasoning here. Max 100 words>')[0])  + '\nResponse:\n'## Specific prompt for question

    output_df.loc[i] = [df['Question ID'].iloc[i], prompts1,  sys_prompt1]

  return output_df

def sat_en_prompts(df):  ## Takes in the csv from GitHub and outputs prompts
  ## Takes in a dataframe in the form:
  ## | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  ## |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |
  ##
  ## Returns a dataframe in the form:
  ## | Question ID | Full Prompt 1 | Full Prompt 2 |
  ## |     (Int)       |    (Str)      |    (Str)      |
  sys_prompt1 = '''You are a helpful assistant. Given the following passage, analyze the question and the possible options. Then, provide a concise reasoning for what is the best answer. Your reasoning should not exceed 100 words. 
Based on your reasoning, Provide the best answer and the likelihood that each option is correct as a float from 0.0 to 1.0 in a JSON format. The probabilities should sum to 1.0. For example:

Question: <Question>
Options:
A) <Option A>
B) <Option B>
C) <Option C>
D) <Option D>


Response:
{
"Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
"Answer": "<Your answer choice here, as a single letter and nothing else>",
"A": "<Probability choice A is correct. As a float from 0.0 to 1.0>",
"B": "<Probability choice B is correct. As a float from 0.0 to 1.0>",
"C": "<Probability choice C is correct. As a float from 0.0 to 1.0>",
"D": "<Probability choice D is correct. As a float from 0.0 to 1.0>"
}

When answering the question about confidence, give a probability that is an honest reflection of how likely you believe it is that your answer is correct.
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
  header = ['Question ID', 'Full Prompt', 'System Prompt']
  output_df = pd.DataFrame(columns = header)

  #----------------------------------------------------------------------------#

  ## Format questions for benchmark
  letters = ['A', 'B', 'C', 'D']
  options = ['Option A', 'Option B', 'Option C', 'Option D']

  for i in range(len(df)):
    question = df['query'][i].split(' Answer Choices: ')[0]

    sys_prompt_temp1 = sys_prompt1

    option_text = df[options[:num_options]].iloc[i].to_list()
    ## Prompt for specific question
    new_prompt = (sys_prompt_temp1
        .split('Response:')[0]
        .replace('<Question>', question)
        .split('For example:')[1]
        .replace('Question:', 'Passage:')
        .replace('Q: ','\nQuestion: ')
        ) ## Uncomment for Qset with premise.
    
    for j in range(num_options): ## This for loop allows for dynamic question amounts
        new_prompt = new_prompt.replace(f'<Option {letters[j]}>', str(option_text[j]))


    ## Add formatted prompts.
    ## Note that this is formatted to llama so changes may be needed down the line.
    prompts1 = 'Directions:\n' + sys_prompt1 +(new_prompt.split('<Your concise reasoning here. Max 100 words>')[0])  + 'Response:\n'## Specific prompt for question
    prompts1 = prompts1.replace(
            'Question: <Question>',
            '''Passage: <Passage>
Question: <Question>
''')
    output_df.loc[i] = [df['Question ID'].iloc[i], prompts1,  sys_prompt1]

  return output_df
## Map functions to dataset

functions_map = {
    'boolq_valid': boolq_valid_prompts,
    'halu_eval_qa': halu_eval_qa_prompts,
    'life_eval': life_eval_prompts,
    'lsat_ar_test': lsat_ar_test_prompts,
    'sat_en': sat_en_prompts,
    'sciq_test': sciq_test_prompts
}

skip_datasets = ['boolq_valid','halu_eval_qa','life_eval']
skip_datasets = []

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
    prompts = {}
    for df_name, df in question_set_dict.items():
        print(f"Processing {df_name} with formatter...")

        format_function = functions_map[df_name]

        prompts_df = format_function(df)

        prompts[df_name] = prompts_df

    print(f'\nTotal Prompt Datasets Formatted: {len(prompts)}')
    print('------------------------------------------\n')
    return prompts

def save_prompts(prompts_dict, folder_path = 'Prompts/'):
    saved = 0
    for df_name, df in prompts_dict.items():
        path = folder_path + df_name + '_prompts.csv'
        if df is not None:
            print(f'Saving {df_name} to: {path}')
            df.to_csv(path, index = False)
            saved += 1
    print(f'\nTotal Prompt Datasets Saved: {saved}')
    print('------------------------------------------\n')
    
def run_all_batch(debug = True):
    for model_instance in all_models:
        print(f'Submitting Batch Process for {model_instance.name}')
        for qset_name, qset in prompts_dict.items():
            if qset_name in skip_datasets: continue # Skips Dataset

            try:
                if debug:
                    prompts_df = qset.iloc[:3]
                else:
                    prompts_df = qset
                
                system = prompts_df['System Prompt'].iloc[0]
                
                # Since each model in all_models is an instance of a model class,
                # we can get the model name from the instance's name attribute.
                model_name = model_instance.name

                model_instance.run_batch(
                    prompts=prompts_df,
                    model=model_name,
                    system_prompt=system,
                    dataset_name=qset_name
                )

                print(f'    {qset_name} Submitted ✅')
            except Exception as e:
                print(f'    Error completing batch request for {model_instance.name} on {qset_name}:\n{e}')
            if debug:
                return 0

            


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
models = {
    'GPT': {
        'class': GPTModels,
        'api_key_name': 'OPENAI_API_KEY',  # Environment variable for the API key
        'models': [
            #'gpt-4o',
            'o3-2025-04-16'
        ]
    },
    'Claude': {
        'class': ClaudeModels,
        'api_key_name': 'ANTHROPIC_API_KEY',  # Environment variable for the API key
        'models': [
            #'claude-3-7-sonnet-20250219',
            #'claude-3-haiku-20240307'
            #'claude-sonnet-4-20250514'
        ]
    },
    'Gemini': {
        'class': GeminiModels,
        'api_key_name': 'GOOGLE_API_KEY',  # Environment variable for the API key
        'models': [
            #'gemini-2.5-flash',
            #'gemini-2.5-pro'
        ]
    }
}



if __name__ == '__main__':
    # Example of how to use the init_models function
    # Make sure your .env file has OPENAI_API_KEY, ANTHROPIC_API_KEY, and GOOGLE_API_KEY

    all_models = init_models(models)
    #print("Returned model instances:", all_models)

    # Example of how to use the import_datasets function
    all_datasets = import_datasets()
    print("\nAvailable datasets:", list(all_datasets.keys()))

    prompts_dict = format_prompts(all_datasets)

    save_prompts(prompts_dict)

    run_all_batch(debug = False)
