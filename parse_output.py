import pandas as pd
import numpy as np
from pprint import pprint
import json
import re
import ast

file_path = "C:\Users\Noam Michael\Desktop\Comparing-Confidence-in-LLMs\Raw Results\gpt-4o\batch_6882b9e8c0b88190a6078d1259b0bbac_output.jsonl"


# Load all lines from the file
entries = []
with open(file_path, "r") as f:
    for line in f:
        entries.append(json.loads(line))

# GPT PARSER

answer_list = []
answer_index_list = []
answer_token_list = []
answer_token_logprobs_list = []

t1 = []
t2 = []
t3 = []
t4 = []
t5 = []

t1_probs = []
t2_probs = []
t3_probs = []
t4_probs = []
t5_probs = []

correct_format = []
coerce = []

for entry in entries:
  response_tokens = entry['response']['body']['choices'][0]['logprobs']['content']
  content = entry['response']['body']['choices'][0]['message']['content']


  # Get Answer and Answer Index
  try:
    answer = ast.literal_eval(content)['Answer']
    correct_format.append(True)
    coerce.append(False)
  except:
    #print('Old Content')
    #print(content)

    ## Fix All possible issues with content:
    try:
      new_content = (content
                .replace('Response:', '')
                .replace(':"', '":') ## Update this for new format with "
                .strip()
      )
      #print('New Content')
      #print(new_content)
      answer = ast.literal_eval(new_content)['Answer']
      correct_format.append(False)
      coerce.append(True)
    except:
      ## If Uncoerceable
      coerce.append(False)
      correct_format.append(False)
      answer_list.append(None)
      answer_index_list.append(None)
      answer_token_list.append(None)
      answer_token_logprobs_list.append(None)

      t1.append(None)
      t2.append(None)
      t3.append(None)
      t4.append(None)
      t5.append(None)

      t1_probs.append(None)
      t2_probs.append(None)
      t3_probs.append(None)
      t4_probs.append(None)
      t5_probs.append(None)
      continue




  answer_list.append(answer)
  pattern = r'"(' + re.escape(answer) + r')"'
  match = re.search(pattern, content)

  #answer_index = content.find(str(answer))
  if match is None:
    print(content)
    continue
  answer_index = match.start() #+ 1
  answer_index_list.append(answer_index)

  #print(f'Answer: {answer:<10} | Answer Index: {answer_index}')

  # Find answer token in JSON
  position = 0
  str_char = 0
  while str_char < answer_index:
    token_info = response_tokens[position]
    str_char += len(token_info['bytes'])
    position += 1

  answer_token = response_tokens[position]['token']
  answer_token_logpobs = response_tokens[position]['top_logprobs']

  tokens = []
  logprobs = []
  for token in answer_token_logpobs:
    tokens.append(token['token'])
    try:
      logprobs.append(token['logprob'])
    except:
      print(token)
      logprobs.append(0)

  probs = np.exp(logprobs)
  probs = logprobs

  t1.append(tokens[0])
  t2.append(tokens[1])
  t3.append(tokens[2])
  t4.append(tokens[3])
  t5.append(tokens[4])

  t1_probs.append(probs[0])
  t2_probs.append(probs[1])
  t3_probs.append(probs[2])
  t4_probs.append(probs[3])
  t5_probs.append(probs[4])


  answer_token_list.append(answer_token)
  answer_token_logprobs_list.append(answer_token_logpobs)

# Make into dataframe]

data = {
  'answer': answer_list,
  'token_index': answer_index_list,
  'token': answer_token_list,
  't1': t1,
  't1_prob': t1_probs,
  't2': t2,
  't2_prob': t2_probs,
  't3': t3,
  't3_prob': t3_probs,
  't4': t4,
  't4_prob': t4_probs,
  't5': t5,
  't5_prob': t5_probs,
  'correct_format': correct_format,
  'coerce': coerce
}
print(f'{"Category":<15}| {"Length":<5} |  Mean')
print('-' * 42)
for category in data:
  try:
    print(f'{category:<15}| {len(data[category]):<5}  | {np.mean([float(x) for x in data[category]if x is not None]):.5}')
  except:
    print(f'{category:<15}| {len(data[category]):<5}  |')



df = pd.DataFrame(data)
display(df)