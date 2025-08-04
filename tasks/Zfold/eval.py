import os


from BPfold.util.misc import get_file_name
from BPfold.util.RNA_kit import read_fasta, write_fasta, read_SS

from pred_and_fold import pred_and_fold_all, pred_and_fold_all_with_MSA


def prepare_output_pdb(dest, src):
    for d in os.listdir(src):
        cur_dest = os.path.join(dest, d)
        os.makedirs(cur_dest, exist_ok=True)
        for f in os.listdir(os.path.join(src, d)):
            if f.endswith('.pdb'):
                shutil.copy(os.path.join(src, d, f), os.path.join(cur_dest, f))


if __name__ == '__main__':
    test_trrosettarna_with_MSA = True

    PREFIX = os.path.abspath('.')
    # DATA_DIR = '/data/heqinzhu/gitrepo/RNA3d_test_data'
    DATA_DIR = 'RNA3d_test_data'
    dest = 'output'
    pred_dir = os.path.join(dest, 'pred_results')

    selected_dataset = ['CASP15_RNAs', '20_RNA_Puzzles']

    LM_para_name = 'structRFM_checkpoint'
    Zfold_para_names = [
                        'Zfold_checkpoint',
                       ]
    selected_dataset = [dataset for dataset in os.listdir(DATA_DIR) if selected_dataset and dataset in selected_dataset]

    for Zfold_para_name in Zfold_para_names:
        cur_dest = os.path.join(dest, Zfold_para_name)
        if Zfold_para_name.startswith('tr'):
            stru_feat_type = 'SS'
        elif Zfold_para_name.startswith('both'):
            stru_feat_type = 'both'
        else:
            stru_feat_type = 'LM'

        dataset_nameseqs = {}
        for dataset in selected_dataset:
            name_seq_pairs = []
            cur_data_dir = os.path.join(DATA_DIR, dataset)
            for f in os.listdir(os.path.join(cur_data_dir, 'ss')):
                name = get_file_name(f)
                if f.endswith('.bpseq'):
                    seq, connects = read_SS(os.path.join(cur_data_dir, 'ss', f))
                    name_seq_pairs.append((name, seq))
            dataset_nameseqs[dataset] = name_seq_pairs

        for dataset, name_seq_pairs in dataset_nameseqs.items():
            cur_data_dir = os.path.join(DATA_DIR, dataset)
            print(cur_data_dir, Zfold_para_name, stru_feat_type)
            OUTPUT_DIR = os.path.join(cur_dest, dataset)
            print('begin predicting')
            if test_trrosettarna_with_MSA:
                pred_and_fold_all_with_MSA( 
                                           name_seq_pairs,
                                           OUTPUT_DIR, PREFIX, cur_data_dir, 
                                           rerun=False, fast_test=False,
                                           LM_para_name=LM_para_name, 
                                           Zfold_para_name=Zfold_para_name,
                                           stru_feat_type=stru_feat_type,
                                      )
            else:
                pred_and_fold_all(
                         name_seq_pairs, OUTPUT_DIR, PREFIX, 
                         rerun=False, fast_test=False, 
                         LM_para_name=LM_para_name, Zfold_para_name=Zfold_para_name,
                        )
        prepare_output_pdb(os.path.join(pred_dir, Zfold_para_name), cur_dest)
