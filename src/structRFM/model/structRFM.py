import torch
import torch.nn as nn
from transformers import LlamaConfig, BertConfig
from transformers import LlamaForCausalLM, LlamaModel, BertForMaskedLM

## For long seq
# StripedHyenaConfig(**model_config)
# StripedHyenaModelForCausalLM
# BertForPretraining: model.get_pool_output,  model.get_sequence_output

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


def get_bert(dim, layer, from_pretrained=None, tokenizer=None, model_class=BertForMaskedLM, max_length=514, pretrained_length=None, *args, **kwargs):
    if tokenizer is None:
        from ..data.tokenizer import get_mlm_tokenizer
        tokenizer = get_mlm_tokenizer(max_length=max_length)
    model_config = BertConfig(
         vocab_size=len(tokenizer),
         hidden_size=dim,
         num_hidden_layers=layer,
         num_attention_heads=12,
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
        if pretrained_length and pretrained_length<tokenizer.model_max_length:
            model = model_class(model_config)
            ori_model = model_class.from_pretrained(from_pretrained)
            state_dict = ori_model.state_dict()
            del state_dict["bert.embeddings.position_embeddings.weight"]  # del pos embed weight
            model.load_state_dict(state_dict, strict=False)

            # deal with extended pos emb
            ori_pos_emb = ori_model.bert.embeddings.position_embeddings.weight.data
            new_pos_emb = torch.randn(tokenizer.model_max_length, 768) 
            new_pos_emb[:pretrained_length] = ori_pos_emb  
            model.bert.embeddings.position_embeddings.weight.data = new_pos_emb
            return model
        else:
            return model_class.from_pretrained(from_pretrained, config=model_config)
    else:
        return model_class(config=model_config)


class structRFM(BertForMaskedLM):
    # NOTICE: explicitly define the `labels` para, not rely on kargs, otherwise the `labels` para won't be correctly passed and the model won't return `eval_loss` when saving checkpoint.
    def forward(self, input_ids, attention_mask, labels=None, connects=None, *args, **kargs):
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, *args, **kargs)


def get_structRFM(dim=768, layer=12, from_pretrained=None, tokenizer=None, pretrained_length=None, *args, **kargs):
    return get_bert(dim=dim, layer=layer, from_pretrained=from_pretrained, tokenizer=tokenizer, model_class=structRFM, pretrained_length=pretrained_length, *args, **kargs)


class structRFM_for_cls(nn.Module):
    def __init__(self, num_class, dim=768, layer=12, from_pretrained=None, tokenizer=None, use_mean_feature=False):
        '''
            use mean seq feature instead of cls token for classification.
        '''
        super(structRFM_for_cls, self).__init__()
        self.structRFM = get_structRFM(dim=dim, layer=layer, from_pretrained=from_pretrained, tokenizer=tokenizer, output_hidden_states=True)
        self.cls = nn.Sequential(
                nn.Linear(in_features=dim, out_features=dim),
                nn.GELU(),
                nn.LayerNorm(dim),
                nn.Dropout(0.1),
                nn.Linear(in_features=dim, out_features=num_class),
        )
        self.use_mean_feature = use_mean_feature

    def forward(self, input_ids, attention_mask=None):
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


def get_structRFM_for_cls(num_class, dim=768, layer=12, from_pretrained=None, tokenizer=None, freeze_base=True, use_mean_feature=False, *args, **kargs):
    model = structRFM_for_cls(num_class=num_class, dim=dim, layer=layer, from_pretrained=from_pretrained, tokenizer=tokenizer, use_mean_feature=use_mean_feature, *args, **kargs)
    if freeze_base:
        for name, para in model.structRFM.named_parameters():
            para.requires_grad = False
    return model
