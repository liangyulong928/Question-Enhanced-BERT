import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from QIModel import QIBertModel
from dataSet_process import fewfc_train_set_build
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from torch.optim import AdamW

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained("./chinese-bert-wwm-ext")
n_epoch = 10
batch_size = 12
learning_rate = 5e-5
adam_epsilon = 1e-8
warmup_steps = 0
max_grad_norm = 1.0


def TransferTrain(fewness):
    dataset = fewfc_train_set_build(fewness)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = QIBertModel.from_pretrained("./chinese-bert-wwm-ext")
    if os.path.isfile("output/base_model.pth"):
        model.load_state_dict(torch.load("output/base_model.pth"), False)
    else:
        raise EnvironmentError("Train model is not exist")
    model.to(device)
    t_total = int(n_epoch * len(train_loader))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=learning_rate * t_total,
                                                num_training_steps=t_total)

    print('------------------------model train---------------------------')
    model.train()
    for epoch in range(n_epoch):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        loop.set_description(f'Epoch [{epoch + 1}/{n_epoch}]')
        for step, batch in loop:
            inputs = {'input_ids': batch['input_ids'].to(device),
                      'token_type_ids': batch['token_type_ids'].to(device),
                      'attention_mask': batch['attention_mask'].to(device),
                      'start_positions': batch['answer_start'].to(device),
                      'end_positions': batch['answer_end'].to(device),
                      'sep_index': batch['sep_index'].to(device)}
            output = model(**inputs)
            loss = output[0]
            loss = loss.mean()
            loop.set_postfix(loss=loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

    torch.save(model.state_dict(), f"output/transfer_model_few_{fewness}.pth")


if __name__ == '__main__':
    for i in [100, 50, 20, 10, 4]:
        TransferTrain(i)
