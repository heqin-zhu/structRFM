from src.Zfold.RNAeval.RNAeval import iter_pred_gt_model_dataset, prepare_af3_pred, prepare_output_pdb, cal_all_metrics
from src.Zfold.RNAeval.RNA_assessment import InteractionNetworkFidelity as INF


if __name__ == '__main__':
    synthetic_names = ['R1126', 'R1128', 'R1136', 'R1138']
    metric_names = ['RMSD', 'eRMSD', 'DI', 'INF']
    prefix = 'output'
    pred_pre = os.path.join(prefix, 'pred_results')
    gt_pre = '/public/share/heqinzhu_share/structRFM/rna3d_vis/native'
    out_name = 'RNA3d_metrics.csv'
    all_metric_path =  f'all_{out_name}'

    models = [
              'af3',
              'trRosettaRNA',
              'RiNALMo-Mega',
              'structRFM',
             ]
    for model in models:
        prepare_func = prepare_af3_pred if model=='af3' else prepare_output_pdb
        prepare_func(os.path.join(pred_pre, model), os.path.join(prefix, model))
    all_data = []
    dataset_names = {
                     # 'R1251', 
                     'CASP16': ['R1203', 'R1205', 'R1209', 'R1211', 'R1212', 'R1242', 'R1260', 'R1261', 'R1263', 'R1283v1', 'R1285', 'R1286', 'R1296'], 
                     'CASP15_RNAs': ['R1107', 'R1108', 'R1116', 'R1117', 'R1126', 'R1128', 'R1136', 'R1138', 'R1149', 'R1156', 'R1189', 'R1190'], 
                     '20_RNA_Puzzles': ['PZ10', 'PZ11', 'PZ12', 'PZ13', 'PZ14Bound', 'PZ14Free', 'PZ15', 'PZ17', 'PZ19', 'PZ1', 'PZ20', 'PZ21', 'PZ22', 'PZ23', 'PZ25', 'PZ27', 'PZ29', 'PZ30', 'PZ33', 'PZ5']
                    }
    for dic in iter_pred_gt_model_dataset(pred_pre, gt_pre, models, dataset_names=dataset_names):
        pred_pdb = dic['pred_pdb']
        gt_pdb = dic['gt_pdb']
        metric_data = cal_all_metrics(pred_pdb, gt_pdb, eval_SS=True)
        dic.update(metric_data)
        all_data.append(dic)
    df = pd.DataFrame(all_data)
    df.to_csv(all_metric_path, index=False)
