import pandas as pd
import requests
import json

def terminate(key, instance): ## Terminates the instance
    
    url = "https://cloud.lambda.ai/api/v1/instance-operations/terminate"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f'Bearer {key}'
    }
    payload = {
        "instance_ids": [instance] 
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)
    

    return 0

def get_instances(key, debug = False):
    url = "https://cloud.lambda.ai/api/v1/instances"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f'Bearer {key}'
    }
    
    response = requests.get(url, headers=headers)
    if debug:
        print("Status Code:", response.status_code)
        print("Response Body:", response.text)
    response_dict = json.loads(response.text)

    instance_id_list = []
    names_list = []
    for instance in  response_dict['data']:
        instance_id_list.append(instance['id'])
        names_list.append(instance['name'])
    instance_dictionary = dict(zip(names_list, instance_id_list))
    return instance_dictionary

def get_key(path, debug = False):
    with open(path, 'r') as file:
        key = file.read()
        if debug: print(key)

    return key

def main():
    path_to_key = '<PATH TO YOUR KEY>'
    my_key = get_key(path_to_key)
    my_id = get_instances(key)
    terminate(key = my_key, instance = my_instance)