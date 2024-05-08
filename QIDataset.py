from torch.utils.data import Dataset


class QIDataset(Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask, sep_index, answer_start, answer_end):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.sep_index = sep_index
        self.answer_start = answer_start
        self.answer_end = answer_end

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        token_type_ids = self.token_type_ids[index]
        sep_index = self.sep_index[index]
        answer_start = self.answer_start[index]
        answer_end = self.answer_end[index]

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'sep_index': sep_index,
            'answer_start': answer_start,
            'answer_end': answer_end
        }

    def __len__(self):
        return len(self.input_ids)
