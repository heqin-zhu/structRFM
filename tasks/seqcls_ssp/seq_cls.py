import numpy as np
import torch
from torch import nn


class RNABertForSeqCls(nn.Module):
    def __init__(self, bert, num_classes, hidden_size=120):
        super(RNABertForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def forward(self, input_ids, **kargs):
        _, pooled_output = self.bert(input_ids)
        logits = self.classifier(pooled_output)
        return logits


class RNAMsmForSeqCls(nn.Module):
    def __init__(self, bert, num_classes, hidden_size=768):
        super(RNAMsmForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def forward(self, input_ids, **kargs):
        output = self.bert(input_ids, repr_layers=[10])
        representations = output["representations"][10][:, 0, 0, :]
        logits = self.classifier(representations)
        return logits


class RNAFmForSeqCls(nn.Module):
    def __init__(self, bert, num_classes, hidden_size=640):
        super(RNAFmForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=True)

    def forward(self, input_ids, **kargs):
        output = self.bert(input_ids, repr_layers=[12])
        representations = output["representations"][12][:, 0, :]
        logits = self.classifier(representations)
        return logits


class RiNALMoForSeqCls(nn.Module):
    def __init__(self, bert, num_classes, hidden_size=640): # 480
        super(RiNALMoForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, **kargs):
        output = self.bert(input_ids, output_hidden_states=True, attention_mask=attention_mask)
        embeddings = output.hidden_states[-1][:, 0, :]
        logits = self.classifier(embeddings)
        return logits


class Evo2ForSeqCls(nn.Module):
    def __init__(self, num_classes, hidden_size, max_len=512):
        super(Evo2ForSeqCls, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.evo2_embedding_dic = np.load('/public/share/heqinzhu_share/structRFM/evo2_embeddings/seqcls_train.npz')
        self.evo2_embedding_test_dic = np.load('/public/share/heqinzhu_share/structRFM/evo2_embeddings/seqcls_test.npz')
        self.max_len = max_len

    def get_embedding(self, name):
        if name in self.evo2_embedding_dic:
            return self.evo2_embedding_dic[name]
        elif name in self.evo2_embedding_test_dic:
            return self.evo2_embedding_test_dic[name]
        else:
            raise Exception(f'Unknown name: {name}')

    def forward(self, input_ids, attention_mask=None, **kargs):
        names = kargs['names']
        embeddings = []
        for name in names:
            embed = self.get_embedding(name)
            if embed.shape[1]>self.max_len:
                embed = embed[:, :self.max_len, :]
            else:
                embed = np.pad(embed, ((0, 0),(0, self.max_len-embed.shape[1]),(0, 0)), constant_values=0)
            embed = torch.tensor(embed, device=input_ids.get_device())
            embeddings.append(embed)
        cls_feat = torch.cat(embeddings, dim=0).mean(dim=1)
        logits = self.classifier(cls_feat)
        return logits
