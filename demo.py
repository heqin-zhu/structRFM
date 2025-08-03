# pip install structRFM-0.0.1.tar.gz
# pip install transformers, datasets

from src.structRFM.infer import structRFM_infer


if __name__ == '__main__':
    from_pretrained = '/public/share/heqinzhu_share/structRFM/structRFM_checkpoint' # TODO, update path
    model = structRFM_infer(from_pretrained=from_pretrained, max_length=514)

    # func: unmask
    unmasked_seq, pred_tokens, top_tokens_list = model.unmask('A[mask]GUAGUAGUCCCG[mask]AAUG', top_k=2)


    seq = 'AGUACGUAGUA'
    output_attentions = True

    print('seq len:', len(seq))
    ## (1+L+1)x 768,  [CLS] seq [SEP]
    features, attentions = model.extract_feature(seq, return_all=True, output_attentions=output_attentions)

    # feat  tuple: layer=12, tuple[i]: batch x L x hidden_dim(=768)
    last_feat = features[-1]
    # for classification, use feat[0,:], 1x768
    cls_feat = last_feat[0,:] # 1x768
    # for 1D feature, use feat[1:-1,:], Lx768
    feat1d = last_feat[1:-1, :] # Lx768
    # for 2D feature, use  x @ x.transpose(-1, -2), x = feat[1:-1,:]
    feat2d = feat1d @ feat1d.transpose(-1,-2) # LxL

    print('classification feature:', cls_feat.shape)
    print('sequence feature:', feat1d.shape)
    print('matrix feature:', feat2d.shape)

    # atten   tuple: layer=12, tuple[i]: batch x head(=12) x L x L
    # remove special tokens
    attentions = tuple([atten[:, :, 1:-1, 1:-1] for atten in attentions])
    print('attentions', len(attentions), attentions[0].shape)

    for layer in range(len(attentions)):
        layer_atten = attentions[layer]
        batch, num_head, _, _ = layer_atten.shape
        for head in range(num_head):
            matrix = layer_atten[0,head]

    exit()
    # func: further finetuning
    from transformers import Trainer, TrainingArguments
    from src.structRFM.infer import save_seqs_to_csv
    from src.structRFM.data import preprocess_and_load_dataset

    tag = 'mlm'
    mlm_structure=True
    seqs = ['AUGCNGUAK', 'AAA', 'AUGAGNAGK']
    data_path = 'tmp.csv'
    save_seqs_to_csv(data_path, seqs)
    train_dataset = preprocess_and_load_dataset(data_path, model_mlm.get_tokenizer(), tag, with_structure=mlm_structure, save_to_disk=False)
    training_args = TrainingArguments(
                            output_dir = 'finetune',
                            num_train_epochs = 3,
                     )
    model_mlm.model.train()
    trainner = Trainer(
                       model=model_mlm.model,
                       args=training_args,
                       train_dataset=train_dataset, # TODO
                      )
