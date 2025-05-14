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


def get_bert(dim, layer, from_pretrained=None, tokenizer=None, model_class=BertForMaskedLM, max_length=514, *args, **kwargs):
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
        return model_class.from_pretrained(from_pretrained, config=model_config)
    else:
        return model_class(config=model_config)


class RNAStruBert(BertForMaskedLM):
    # NOTICE: explicitly define the `labels` para, not rely on kargs, otherwise the `labels` para won't be correctly passed and the model won't return `eval_loss` when saving checkpoint.
    def forward(self, input_ids, attention_mask, labels=None, connects=None, *args, **kargs):
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, *args, **kargs)


def get_RNAStruBert(dim=768, layer=12, from_pretrained=None, tokenizer=None, *args, **kargs):
    return get_bert(dim=dim, layer=layer, from_pretrained=from_pretrained, tokenizer=tokenizer, model_class=RNAStruBert, *args, **kargs)


class RNAStruBert_for_cls(nn.Module):
    def __init__(self, num_class, dim=768, layer=12, from_pretrained=None, tokenizer=None):
        super(RNAStruBert_for_cls).__init__()
        self.RNAStruBert = get_RNAStruBert(dim=dim, layer=layer, from_pretrained=from_pretrained, tokenizer=tokenizer, output_hidden_states=True)
        self.cls = nn.Sequential(
                Linear(in_features=dim, out_features=dim),
                nn.GELU(),
                nn.LayerNorm(dim),
                nn.Dropout(0.1),
                nn.Linear(in_features=dim, out_features=num_class),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.RNAStruBert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_hidden = outputs.hidden_states[-1][:, 0, :]
        logits = self.cls(cls_hidden)
        return logits


def get_RNAStruBert_for_cls(num_class, dim=768, layer=12, from_pretrained=None, tokenizer=None, *args, **kargs):
    return RNAStruBert_for_cls(num_class=num_class, dim=dim, layer=layer, from_pretrained=from_pretrained, tokenizer=tokenizer, *args, **kargs)
