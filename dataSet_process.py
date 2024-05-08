import json

import torch

from QIDataset import QIDataset
from utils import convert_file_to_json, token_location_redirection
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("./chinese-bert-wwm-ext")
duee_source_data_path = 'data/DuEE1.0/origin/duee_train.json/duee_train.json'
duee_event_schema_path = 'data/DuEE1.0/process/event_schema.json'
fewfc_source_data_path = 'data/FewFC-main/origin/train_base.json'
fewfc_event_schema_path = 'data/FewFC-main/process/event_schema.json'


def duee_train_set_build():
    input_ids = []
    token_type_ids = []
    attention_mask = []
    answer_start_index = []
    answer_end_index = []
    sep_index = []
    json_list = convert_file_to_json(duee_source_data_path)
    with open(duee_event_schema_path, "r") as infile:
        event_schema = json.load(infile)
    for jsonl in json_list:
        content = tokenizer.tokenize(jsonl['text'])
        for event in jsonl['event_list']:
            argument_index = {}
            arguments = event_schema[event['event_type']]
            for argument in event['arguments']:
                argument_index[argument['role']] = argument['argument']
            for argument in arguments:
                question = f"{event['event_type']}中的角色\"{argument}\"是什么?"[:62]
                question_length = len(tokenizer.tokenize(question))
                token = tokenizer.encode_plus(question, jsonl['text'], max_length=256, padding='max_length')
                word = argument_index.get(argument, None)
                input_ids.append(token['input_ids'])
                token_type_ids.append(token['token_type_ids'])
                attention_mask.append(token['attention_mask'])
                sep_index.append(question_length + 1)
                if word:
                    a, b = token_location_redirection(content, word)
                    answer_start_index.append(a + question_length + 2)
                    answer_end_index.append(b + question_length + 1)
                else:
                    answer_start_index.append(0)
                    answer_end_index.append(0)
    dataset = QIDataset(torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(token_type_ids, dtype=torch.long),
                        torch.tensor(attention_mask, dtype=torch.long),
                        torch.tensor(sep_index, dtype=torch.long).unsqueeze(1),
                        torch.tensor(answer_start_index, dtype=torch.long).unsqueeze(1),
                        torch.tensor(answer_end_index, dtype=torch.long).unsqueeze(1))
    return dataset


def fewfc_train_set_build(fewness):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    answer_start_index = []
    answer_end_index = []
    sep_index = []
    json_list = convert_file_to_json(fewfc_source_data_path)
    with open(fewfc_event_schema_path, "r") as infile:
        event_schema = json.load(infile)
    i = 0
    for jsonl in json_list:
        if i % fewness == 0:
            event_set = set()
            content = tokenizer.tokenize(jsonl['content'])
            for event in jsonl['events']:
                argument_index = {}
                if event['type'] not in event_set and len(event['mentions']) > 1:
                    event_set.add(event['type'])
                    for argument in event['mentions']:
                        if argument['role'] != 'trigger':
                            argument_index[argument['role']] = argument['word']
                    for argument in event_schema[event['type']]:
                        question = f"{event['type']}中的角色\"{event_schema[event['type']][argument]}\"是什么?"[:62]
                        question_length = len(tokenizer.tokenize(question))
                        token = tokenizer.encode_plus(question, jsonl['content'][:191], max_length=256, padding='max_length')
                        word = argument_index.get(argument, None)
                        input_ids.append(token['input_ids'])
                        token_type_ids.append(token['token_type_ids'])
                        attention_mask.append(token['attention_mask'])
                        sep_index.append(question_length + 1)
                        if word:
                            a, b = token_location_redirection(content, word)
                            answer_start_index.append(a + question_length + 2)
                            answer_end_index.append(b + question_length + 1)
                        else:
                            answer_start_index.append(0)
                            answer_end_index.append(0)
        i += 1
    dataset = QIDataset(torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(token_type_ids, dtype=torch.long),
                        torch.tensor(attention_mask, dtype=torch.long),
                        torch.tensor(sep_index, dtype=torch.long).unsqueeze(1),
                        torch.tensor(answer_start_index, dtype=torch.long).unsqueeze(1),
                        torch.tensor(answer_end_index, dtype=torch.long).unsqueeze(1))
    return dataset
