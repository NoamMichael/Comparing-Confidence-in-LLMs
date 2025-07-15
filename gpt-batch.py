from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

path = 'Batches/test-batch-requests.jsonl'
batch_input_file = client.files.create(
    file=open(path, "rb"),
    purpose="batch"
)

print('Made input')
print(batch_input_file)

batch_input_file_id = batch_input_file.id
client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "test job"
    }
)

print('created batch')
print(batch_input_file_id)