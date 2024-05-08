import csv
import json
import os

import torch

from QIModel import QIBertModel
from dataSet_process import duee_event_schema_path, fewfc_event_schema_path
from utils import convert_file_to_json, tokenizer, token_location_redirection

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def duee_eval():
    model = QIBertModel.from_pretrained("chinese-bert-wwm-ext")
    if os.path.isfile("output/base_model.pth"):
        model.load_state_dict(torch.load("output/base_model.pth"), False)
    else:
        raise EnvironmentError("Train model is not exist")
    model.to(device)
    print('------------------------model evaluation----------------------')
    json_list = convert_file_to_json("data/DuEE1.0/origin/duee_dev.json/duee_dev.json")
    with open(duee_event_schema_path, "r") as infile:
        event_schema = json.load(infile)
    data = []
    acc_num = 0
    predict_num = 0
    golden_num = 0
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
                token = tokenizer.encode_plus(question, jsonl['text'], max_length=256, padding='max_length',
                                              return_tensors='pt')
                word = argument_index.get(argument, None)
                input_token = {'input_ids': token['input_ids'].to(device),
                               'token_type_ids': token['token_type_ids'].to(device),
                               'attention_mask': token['attention_mask'].to(device),
                               'sep_index': torch.tensor([[question_length + 1]]).to(device)}
                output = model(**input_token)
                predict_start = torch.argmax(output['start_logits']).item() - question_length - 2
                predict_end = torch.argmax(output['end_logits']).item() - question_length - 1
                if predict_end >= predict_start >= 0:
                    predict_num += 1
                if word:
                    golden_num += 1
                    a, b = token_location_redirection(content, word)
                    if predict_start <= a <= predict_end:
                        acc_num += 1
                    row = [question, jsonl['text'], word,
                           ''.join(content[predict_start: predict_end]),
                           a, predict_start, predict_end]
                else:
                    row = [question, jsonl['text'], word,
                           ''.join(content[predict_start: predict_end]),
                           -1, predict_start, predict_end]
                print(row)
                data.append(row)
    precision = acc_num / predict_num
    recall = acc_num / golden_num
    print(f'precision = {precision} , recall = {recall} , f1 = {2 * (precision * recall) / (precision + recall)}')
    with open(f'junit_duee_eval.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def transfer_eval(fewness):
    model = QIBertModel.from_pretrained("chinese-bert-wwm-ext")
    if os.path.isfile(f"output/transfer_model_few_{fewness}.pth"):
        model.load_state_dict(torch.load(f"output/transfer_model_few_{fewness}.pth"), False)
    else:
        raise EnvironmentError("Train model is not exist")
    model.to(device)
    print('------------------------model evaluation----------------------')
    json_list = convert_file_to_json("data/FewFC-main/origin/test_base.json")
    with open(fewfc_event_schema_path, "r") as infile:
        event_schema = json.load(infile)
    data = []
    acc_num = 0
    predict_num = 0
    golden_num = 0
    for jsonl in json_list:
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
                    input_token = {'input_ids': token['input_ids'].to(device),
                                   'token_type_ids': token['token_type_ids'].to(device),
                                   'attention_mask': token['attention_mask'].to(device),
                                   'sep_index': torch.tensor([[question_length + 1]]).to(device)}
                    output = model(**input_token)
                    predict_start = torch.argmax(output['start_logits']).item() - question_length - 2
                    predict_end = torch.argmax(output['end_logits']).item() - question_length - 1
                    if predict_end >= predict_start >= 0:
                        predict_num += 1
                    if word:
                        golden_num += 1
                        a, b = token_location_redirection(content, word)
                        if predict_start <= a <= predict_end:
                            acc_num += 1
                        row = [question, jsonl['content'], word,
                               ''.join(content[predict_start: predict_end]),
                               a, predict_start, predict_end]
                    else:
                        row = [question, jsonl['content'], word,
                               ''.join(content[predict_start: predict_end]),
                               -1, predict_start, predict_end]
                    data.append(row)
    precision = acc_num / predict_num
    recall = acc_num / golden_num
    print(f'precision = {precision} , recall = {recall} , f1 = {2 * (precision * recall) / (precision + recall)}')
    with open(f'junit_duee_eval.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


if __name__ == '__main__':
    for i in [100, 50, 20, 10, 4]:
        transfer_eval(i)
