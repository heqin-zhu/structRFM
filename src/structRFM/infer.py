import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import get_structRFM
from .data.tokenizer import get_mlm_tokenizer
from .data.RNAdata import preprocess_and_load_dataset, process_mlm_input_seq


def save_seqs_to_csv(path, seqs, names=None):
    if names is None:
        names = [f'seq{i}' for i in range(len(seqs))]
    df = pd.DataFrame({'name': names, 'seq': seqs})
    df.to_csv(path, index=False)


class structRFM_infer:
    def __init__(self, from_pretrained, max_length=514, dim=768, layer=12, output_hidden_states=True, device=None):
        # set output_hidden_states=True to get the hidden states (features)

        # self.tokenizer = AutoTokenizer(from_pretrained)
        # self.model = AutoModel.from_pretrained(from_pretrained, output_hidden_states=True) # why won't output logtits?
        self.tokenizer = get_mlm_tokenizer(max_length=max_length)
        self.model = get_structRFM(dim=dim, layer=layer, from_pretrained=from_pretrained, tokenizer=self.tokenizer, output_hidden_states=output_hidden_states)
        if device is None:
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model.to(device)
        print(f'Running on {self.model.device}')


    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def unmask(self, masked_seq, top_k=1):
        '''
            model.output.logtis
        '''
        self.model.eval()
        text = process_mlm_input_seq(masked_seq)
        inputs = self.tokenizer(text, return_tensors='pt')
        mask_positions = torch.where(inputs['input_ids'][0] == self.tokenizer.mask_token_id)[0]

        with torch.no_grad(), torch.cuda.amp.autocast():
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
    def model_forward(self, seq, return_inputs=False, is_cal_loss=False, output_attentions=False, is_training=False):
        '''
            Get model outputs.
            seq: str
                seq_len
            return outputs, (inputs): dict, (Tensor)
                {'logits': logits, 'hidden_states': hidden_states', 'attentions': attentions}
        '''
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        text = process_mlm_input_seq(seq)
        inputs = self.tokenizer(text, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        attention = None
        outputs = self.model(**inputs, labels=inputs['input_ids'], output_attentions=output_attentions) if is_cal_loss else self.model(**inputs, output_attentions=output_attentions)
        if return_inputs:
            return outputs, inputs
        else:
            return outputs


    def extract_feature(self, seq, include_special=True, return_all=False, output_attentions=False, is_training=False):
        '''
            Extract hidden feature of input seq.
            seq: str
                seq_len
            return: Tensor
               (all_layers=13) x seq_len x hidden_size768
        '''
        attention = None
        outputs = self.model_forward(seq, output_attentions=True, is_training=is_training)
        hidden_states = outputs.hidden_states  # tuple(B x (seq_len+2) x hidden_size768), tuple_len = (1+layer12) 
        ## batch_size = 1, use the first one 0 of this batch
        sls = slice(None, None) if include_special else slice(1, -1)
        out_features = [hidd[0, sls, :] for hidd in hidden_states] # tuple(seq_len x hidden_size), tuple_len = (1+layer12)
        final_out_features = out_features if return_all else out_features[-1]
        out_attentions = outputs.attentions if return_all else outputs.attentions[-1]
        if output_attentions:
            return final_out_features, out_attentions
        else:
            return final_out_features
