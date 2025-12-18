import torch
import torch.nn as nn
from transformers import LlamaConfig, BertConfig
from transformers import LlamaForCausalLM, LlamaModel, BertForMaskedLM

## For long seq
# StripedHyenaConfig(**model_config)
# StripedHyenaModelForCausalLM
# BertForPretraining: model.get_pool_output,  model.get_sequence_output


def get_model_scale(scale):
    scale_dic = {
             'base': dict(dim=768, layer=12, num_attention_heads=12),
             'large': dict(dim=1024, layer=24, num_attention_heads=16),
           }
    return scale_dic.get(scale, {})


def get_llama_model(dim, layer, from_pretrained, tokenizer, model_class=LlamaModel):
    model_config = LlamaConfig(
        vocab_size=len(tokenizer),
        n_positions=tokenizer.model_max_length,
        hidden_size=dim,
        intermediate_size=dim*4,
        num_hidden_layers=layer,
        num_key_value_heads=16,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_flash_attention_2=True
    )
    if from_pretrained:
        return model_class.from_pretrained(from_pretrained, config=model_config)
    else:
        return model_class(config=model_config)


def get_llama_causal_model(dim, layer, from_pretrained, tokenizer):
    # class myLlamaCausal(nn.Module):
    #     def __init__(self, config):
    #         super().__init__()
    #         self.llama = get_llama_model(dim, layer, from_pretrained, tokenizer)
    #         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    ## out logits: batch x length x vocab_size
    return get_llama_model(dim, layer, from_pretrained, tokenizer, model_class=LlamaForCausalLM)


def get_bert(dim, layer, num_attention_heads=12, from_pretrained=None, max_length=514, tokenizer=None, model_class=BertForMaskedLM, *args, **kwargs):
    if tokenizer is None:
        from ..data.tokenizer import get_mlm_tokenizer
        tokenizer = get_mlm_tokenizer(max_length=max_length)
    model_config = BertConfig(
         vocab_size=len(tokenizer),
         hidden_size=dim,
         num_hidden_layers=layer,
         num_attention_heads=num_attention_heads,
         type_vocab_size=2,
         intermediate_size=dim*4,
         hidden_act="gelu",
         hidden_dropout_prob=0.1,
         attention_probs_dropout_prob=0.1,
         max_position_embeddings=tokenizer.model_max_length,
         initializer_range=0.02,
         *args,
         **kwargs,
    )
    if from_pretrained:
        ori_model = model_class.from_pretrained(from_pretrained)
        ori_pos_emb = ori_model.bert.embeddings.position_embeddings.weight.data
        pretrained_length = ori_pos_emb.shape[0]
        if pretrained_length>tokenizer.model_max_length:
            raise Exception(f'[Error]: Pre-trained length: {pretrained_length}> max_length: {tokenizer.model_max_length}')
        elif pretrained_length<tokenizer.model_max_length:
            model = model_class(model_config)
            print(f'[Warning]: Loading model from "{from_pretrained}": Pre-trained length={pretrained_length}, adaptating on long sequences: {tokenizer.model_max_length}')
            state_dict = ori_model.state_dict()
            del state_dict["bert.embeddings.position_embeddings.weight"]  # del pos embed weight
            model.load_state_dict(state_dict, strict=False)

            # deal with extended pos emb
            new_pos_emb = torch.randn(tokenizer.model_max_length, dim) 
            new_pos_emb[:pretrained_length] = ori_pos_emb  
            model.bert.embeddings.position_embeddings.weight.data = new_pos_emb
            return model
        else:
            print(f'Loading model from "{from_pretrained}", length={tokenizer.model_max_length}')
            return model_class.from_pretrained(from_pretrained, config=model_config)
    else:
        return model_class(config=model_config)


class structRFM(BertForMaskedLM):
    # NOTICE: explicitly define the `labels` para, not rely on kargs, otherwise the `labels` para won't be correctly passed and the model won't return `eval_loss` when saving checkpoint.
    def forward(self, input_ids, attention_mask, labels=None, connects=None, *args, **kargs):
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, *args, **kargs)


def get_structRFM(dim=768, layer=12, num_attention_heads=12, from_pretrained=None, max_length=514, tokenizer=None, *args, **kargs):
    return get_bert(dim=dim, layer=layer, num_attention_heads=num_attention_heads, from_pretrained=from_pretrained, max_length=max_length, tokenizer=tokenizer, model_class=structRFM, *args, **kargs)


class structRFM_for_cls(nn.Module):
    def __init__(self, num_class, dim=768, layer=12, num_attention_heads=12, from_pretrained=None, max_length=514, tokenizer=None, use_mean_feature=False, freeze_base=False, *args, **kargs):
        super(structRFM_for_cls, self).__init__()
        self.structRFM = get_structRFM(dim=dim, layer=layer, num_attention_heads=num_attention_heads, from_pretrained=from_pretrained, max_length=max_length, tokenizer=tokenizer, output_hidden_states=True)
        self.cls = nn.Sequential(
                nn.Linear(in_features=dim, out_features=dim),
                nn.GELU(),
                nn.LayerNorm(dim),
                nn.Dropout(0.1),
                nn.Linear(in_features=dim, out_features=num_class),
        )
        self.use_mean_feature = use_mean_feature
        if freeze_base:
            for name, para in self.structRFM.named_parameters():
                para.requires_grad = False


    def forward(self, input_ids, attention_mask=None, *args, **kargs):
        outputs = self.structRFM(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        if self.use_mean_feature:
            cls_hidden = outputs.hidden_states[-1][:, 1:-1, :].mean(dim=1)
        else:
            cls_hidden = outputs.hidden_states[-1][:, 0, :]
        logits = self.cls(cls_hidden)
        return logits


class structRFM_for_longseq_cls(nn.Module):
    def __init__(self, num_class, dim=768, layer=12, num_attention_heads=12, from_pretrained=None, tokenizer=None, use_mean_feature=False, window_size=512, use_overlapping_window=False, freeze_base=False, *args, **kargs):
        super(structRFM_for_longseq_cls, self).__init__()
        assert ((window_size-1) & window_size)==0, f'window size ({window_size}) must be the power of 2'
        self.structRFM = get_structRFM(dim=dim, layer=layer, num_attention_heads=num_attention_heads, from_pretrained=from_pretrained, max_length=514, tokenizer=tokenizer, output_hidden_states=True)
        self.cls = nn.Sequential(
                nn.Linear(in_features=dim, out_features=dim),
                nn.GELU(),
                nn.LayerNorm(dim),
                nn.Dropout(0.1),
                nn.Linear(in_features=dim, out_features=num_class),
        )
        self.use_mean_feature = use_mean_feature
        self.window_size = window_size
        self.step_size = window_size//2 if use_overlapping_window else window_size
        self.pool = nn.AdaptiveAvgPool1d(1)
        if freeze_base:
            for name, para in self.structRFM.named_parameters():
                para.requires_grad = False


    def get_cls_feat(self, outputs):
        if self.use_mean_feature:
            return outputs.hidden_states[-1][:, 1:-1, :].mean(dim=1)
        else:
            return outputs.hidden_states[-1][:, 0, :]

    def forward(self, input_ids, attention_mask=None, *args, **kargs):
        B, L = input_ids.shape
        begin_id = input_ids[:, 0:1]
        end_id = input_ids[:, -1:0]
        cls_feat_list = []
        for i in range(1, L, self.step_size):
            i_end = min(i+self.window_size, L)
            cur_ids = torch.cat([begin_id, input_ids[:, i:i_end], end_id], dim=1)
            cur_mask = attention_mask
            if cur_mask is not None:
                cur_mask = torch.cat([cur_mask[0], cur_mask[i:i_end], cur_mask[-1]], dim=0)
            outputs = self.structRFM(input_ids=cur_ids, attention_mask=cur_mask)
            cls_feat_list.append(self.get_cls_feat(outputs))
        agg_cls_feat = self.pool(torch.stack(cls_feat_list, dim=-1)).squeeze(-1)
        logits = self.cls(agg_cls_feat)
        return logits
