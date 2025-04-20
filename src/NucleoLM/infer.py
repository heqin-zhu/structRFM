import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import get_bert_mlm_stru_pretraining
from .data.tokenizer import get_mlm_tokenizer
from .data.RNAdata import preprocess_and_load_dataset, process_mlm_input_seq


def save_seqs_to_csv(path, seqs, names=None):
    if names is None:
        names = [f'seq{i}' for i in range(len(seqs))]
    df = pd.DataFrame({'name': names, 'seq': seqs})
    df.to_csv(path, index=False)


class RNALM_MLM:
    def __init__(self, from_pretrained, max_length=514, dim=768, layer=12, output_hidden_states=True):
        self.tokenizer = get_mlm_tokenizer(max_length=max_length)
        # set output_hidden_states=True to get the hidden states (features)
        self.model = get_bert_mlm_stru_pretraining(dim=dim, layer=layer, from_pretrained=from_pretrained, tokenizer=self.tokenizer, output_hidden_states=output_hidden_states)
        if torch.cuda.is_available():
            self.model.cuda()
        print(f'Running on {self.model.device}')

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def unmask(self, masked_seq, top_k=1):
        self.model.eval()
        text = process_mlm_input_seq(masked_seq)
        inputs = self.tokenizer(text, return_tensors='pt')
        mask_positions = torch.where(inputs['input_ids'][0] == self.tokenizer.mask_token_id)[0]

        with torch.no_grad():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        logits = outputs.logits # B x (seq_len+2) x vocab
        predicted_tokens = []
        top_tokens_list = []
        for pos in mask_positions:
            logit = logits[0, pos]
            # predicted_id = logit.argmax().item()
            top_ids = logit.topk(top_k).indices.tolist()
            predicted_token = self.tokenizer.decode(top_ids[0])
            top_tokens_list.append(self.tokenizer.convert_ids_to_tokens(top_ids))
            predicted_tokens.append(predicted_token)

        parts = masked_seq.upper().split(self.tokenizer.mask_token)
        new_parts = []
        for part, pred in zip(parts, predicted_tokens):
            new_parts.append(part)
            new_parts.append(pred)
        new_parts.append(parts[-1])
        return ''.join(new_parts), predicted_tokens, top_tokens_list


    @torch.no_grad()
    def model_forward(self, seq, return_inputs=False):
        '''
            Get model outputs.
            seq: str
                seq_len
            return outputs: dict
                {'logits': logits, 'hidden_states': hidden_states'}
        '''
        self.model.eval()
        text = process_mlm_input_seq(seq)
        inputs = self.tokenizer(text, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        if return_inputs:
            return outputs, inputs
        return outputs


    def extract_feature(self, seq, return_all=False):
        '''
            Extract hidden feature of input seq.
            seq: str
                seq_len
            return: Tensor
               (all_layers=13) x seq_len x hidden_size768
        '''
        outputs = self.model_forward(seq)
        hidden_states = outputs.hidden_states  # tuple(B x (seq_len+2) x hidden_size768), tuple_len = (1+layer12) 
        out_features = [hidd[0,1:-1,:] for hidd in hidden_states] # tuple(seq_len x hidden_size), tuple_len = (1+layer12)
        return out_features if return_all else out_features[-1]
