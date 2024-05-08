from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import torch.nn.functional as F


class QIBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)

        self.attention = nn.Linear(config.hidden_size * 3, 256)
        self.qa_outputs = nn.Linear(config.hidden_size * 2, config.num_labels)

        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            sep_index: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]  # 8 * 256 * 768
        content_pooling, _ = torch.max(hidden_state, dim=1)  # 8 * 768
        sep_index = sep_index.int().tolist()

        pooling_hidden_list = []
        for i in range(hidden_state.size(0)):
            if 0 < sep_index[i][0] < 255:
                pooling = F.max_pool1d(hidden_state[i][:sep_index[i][0]].permute(1, 0),
                                       kernel_size=sep_index[i][0],
                                       stride=sep_index[i][0]).squeeze(-1)
            else:
                pooling = content_pooling[i]
            pooling_hidden_list.append(pooling)
        question_hidden_pooling = torch.stack(pooling_hidden_list, dim=1).permute(1, 0)  # 8 * 768
        intern_pooling = torch.concat([content_pooling, question_hidden_pooling], dim=-1)  # 8 * (768 * 2)

        length_list = []
        for i in range(hidden_state.size(0)):
            if 0 < sep_index[i][0] < 255:
                context_hidden = hidden_state[i][sep_index[i][0] + 1:]

            else:
                context_hidden = hidden_state[i]
            question_intern = torch.concat([context_hidden,
                                            intern_pooling[i].expand(context_hidden.size(0),
                                                                     context_hidden.size(1) * 2)], dim=-1)
            length_intern = self.attention(question_intern)
            length_intern, _ = torch.max(length_intern, dim=0)
            length_list.append(length_intern)
        length_tensor = torch.stack(length_list, dim=1).permute(1, 0)              # 8 * 256
        attention = F.normalize(length_tensor, p=1, dim=1)

        attention_hidden = attention.unsqueeze(1).permute(0, 2, 1) * hidden_state
        sequence_output = torch.cat((hidden_state, attention_hidden), dim=2)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
