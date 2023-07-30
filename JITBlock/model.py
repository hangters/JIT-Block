import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from transformers import RobertaForSequenceClassification


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.manual_dense = nn.Linear(config.feature_size, config.hidden_size * args.max_changed_block_unit)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.liner = nn.Linear(config.hidden_size*args.max_changed_block_unit*2,config.hidden_size)
        self.out_proj_new = nn.Linear(config.hidden_size, 1)

    def forward(self, features, manual_features, **kwargs):
        y = manual_features.float()  # [bs, feature_size]
        y = self.manual_dense(y)
        y = torch.tanh(y)

        x = torch.cat((features, y), dim=-1)
        x = self.dropout(x)
        x = self.liner(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj_new(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.config = config
        self.classifier = RobertaClassificationHead(config, args)
        self.args = args

    def forward(self, inputs_ids, attns, manual_features=None,
                labels=None, weight_dict=None, output_attentions=None):

        outputs = []
        last_layer_attn_weights=[]
        for i in range(inputs_ids.shape[1]):
            output = self.encoder(input_ids=inputs_ids[:, i, :], attention_mask=attns[:, i, :],
                                  output_attentions=output_attentions)
            last_layer_attn_weights.append(output.attentions[self.config.num_hidden_layers - 1][:, :,
                                      0].detach() if output_attentions else None)
            output = torch.tanh(output[0][:, 0, :])
            outputs.append(output)
        outputs = torch.cat(outputs, dim=-1)
        if output_attentions:
            last_layer_attn_weights=torch.cat(last_layer_attn_weights,dim=-1)
        logits = self.classifier(outputs, manual_features)
        prob = torch.sigmoid(logits)
        if labels is not None:
            loss_fct = BCELoss()
            weight_tensor = self.get_loss_weight(labels, weight_dict)
            loss_fct.weight = weight_tensor
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob,last_layer_attn_weights
        else:
            return prob

    def get_loss_weight(self,labels, weight_dict):
        label_list = labels.cpu().numpy().squeeze().tolist()
        weight_list = []

        for lab in label_list:
            if lab == 0:
                weight_list.append(weight_dict['clean'])
            else:
                weight_list.append(weight_dict['defect'])

        weight_tensor = torch.tensor(weight_list).reshape(-1, 1).cuda()
        return weight_tensor
