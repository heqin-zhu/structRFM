from src.structRFM.infer import structRFM_infer


if __name__ == '__main__':
    # Usage examples
    from_pretrained = '/public/share/heqinzhu_share/structRFM/structRFM_checkpoint' # TODO, update path
    model = structRFM_infer(from_pretrained=from_pretrained, max_length=514)

    # unmask seq
    unmasked_seq, pred_tokens, top_tokens_list = model.unmask('A[mask]GUAGUAGUCCCG[mask]AAUG', top_k=2)

    # extract feature
    seq = 'AGUACGUAGUA'
    print('seq len:', len(seq))
    feat_dic = model.extract_feature(seq)
    for k, v in feat_dic.items():
        print(k, v.shape)

    # extract attention: atten   tuple: layer=12, tuple[i]: batch x head(=12) x L x L
    features, attentions = model.extract_raw_feature(seq, return_all=True, output_attentions=True)
    attentions = tuple([atten[:, :, 1:-1, 1:-1] for atten in attentions]) # remove special tokens
    print('attentions', len(attentions), attentions[0].shape)
    for layer in range(len(attentions)):
        layer_atten = attentions[layer]
        batch, num_head, _, _ = layer_atten.shape
        for head in range(num_head):
            matrix = layer_atten[0,head]

    exit()
    # further finetuning
    from transformers import Trainer, TrainingArguments
    from src.structRFM.infer import save_seqs_to_csv
    from src.structRFM.model import get_structRFM
    from src.structRFM.data import preprocess_and_load_dataset, get_mlm_tokenizer

    tokenizer = get_mlm_tokenizer(max_length=514)
    model = get_structRFM(dim=768, layer=12, from_pretrained=from_pretrained, pretrained_length=None, max_length=514, tokenizer=tokenizer)

    tag = 'mlm'
    mlm_structure=True
    seqs = ['AUGCNGUAK', 'AAA', 'AUGAGNAGK']
    data_path = 'tmp.csv'
    save_seqs_to_csv(data_path, seqs)
    train_dataset = preprocess_and_load_dataset(data_path, tokenizer, tag, with_structure=mlm_structure, save_to_disk=False)
    training_args = TrainingArguments(
                            output_dir = 'finetune',
                            num_train_epochs = 3,
                     )
    model.train()
    trainner = Trainer(
                       model=model,
                       args=training_args,
                       train_dataset=train_dataset, # TODO
                      )
