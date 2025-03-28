from transformers import Trainer, TrainingArguments

from src.NucleoLM.infer import RNALM_MLM, save_seqs_to_csv
from src.NucleoLM.data import get_finetune_dataset


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
    from_pretrained = '/heqinzhu/runs/mlm_768x12_lr0.0001/checkpoint-1124126'
    model_mlm = RNALM_MLM(from_pretrained=from_pretrained, max_length=514)

    # func: unmask
    unmasked_seq, pred_tokens, top_tokens_list = model_mlm.unmask('A[mask]GUAGUAGUCCCG[mask]AAUG', top_k=2)

    # func: extract_feature
    # B x (seq_len+2) x hidden_size768
    last_hidden_state = model_mlm.extract_feature('AGUAGUAGUCCCGUG', return_all=False)

    # func: further finetuning
    tag = 'mlm'
    seqs = ['AUGCNGUAK', 'AAA', 'AUGAGNAGK']
    data_path = 'tmp.csv'
    save_seqs_to_csv(data_path, seqs)
    train_dataset = get_finetune_dataset(data_path, model_mlm.get_tokenizer(), tag, save_to_disk=False)
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

