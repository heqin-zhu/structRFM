from transformers import Trainer, TrainingArguments

from src.SgRFM.infer import SgRFM_infer, save_seqs_to_csv
from src.SgRFM.data import preprocess_and_load_dataset


if __name__ == '__main__':
    '''
    BPfold 7M
    LLaMA:
    (dim, layer), para
    (256, 4), 3M
    (384，6), 13M
    (512, 8), 31M
    (768, 12), 106M
    (4096, 32), 1B
    '''

    # model init
    from_pretrained = '/public/share/heqinzhu_share/SgRFM/checkpoint-1174460'
    model_mlm = SgRFM_infer(from_pretrained=from_pretrained, max_length=514)

    # func: unmask
    unmasked_seq, pred_tokens, top_tokens_list = model_mlm.unmask('A[mask]GUAGUAGUCCCG[mask]AAUG', top_k=2)

    # func: extract_feature
    # seq_len x hidden_size768
    last_feature = model_mlm.extract_feature('AGUAGUAGUCCCGUG', include_special=False, return_all=False)
    print('extracted features:', last_feature.shape)
    exit()

    # func: further finetuning
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
