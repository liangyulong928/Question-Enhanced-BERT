import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("./chinese-bert-wwm-ext")


def convert_file_to_json(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    json_list = []
    for line in lines:
        line = line.strip()
        json_obj = json.loads(line)
        json_list.append(json_obj)
    return json_list


def token_location_redirection(token_list, trigger):
    trigger_token = tokenizer.tokenize(trigger)
    for i in range(len(token_list) - len(trigger_token) + 1):
        if token_list[i:i + len(trigger_token)] == trigger_token:
            return i, i + len(trigger_token)
    return 0, 0
