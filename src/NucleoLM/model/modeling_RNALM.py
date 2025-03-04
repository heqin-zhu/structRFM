from transformers import LlamaConfig, BertConfig
from transformers import LlamaForCausalLM, LlamaModel, BertModel

## For long seq
# StripedHyenaConfig(**model_config)
# StripedHyenaModelForCausalLM


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


def get_bert_model(dim, layer, from_pretrained, tokenizer):
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
         initializer_range=0.02
    )
    model_class = BertModel
    if from_pretrained:
        return model_class.from_pretrained(from_pretrained, config=model_config)
    else:
        return model_class(config=model_config)
    # model.get_pool_output,  model.get_sequence_output


class CustomBertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 2)  # 二分类任务

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 取[CLS]向量
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def get_custom_bert():
    config = BertConfig.from_pretrained("bert-base-uncased")
    model = CustomBertClassifier(config)
    # 替换第3层的自注意力
    # layer_index = 2  # 第3层（索引从0开始）
    # model.bert.encoder.layer[layer_index].attention.self = CustomAttention(config)
    return model
