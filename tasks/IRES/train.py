import dgl
import numpy as np
import os
import sys
import time
import random
import argparse, json
import torch
import torch.optim as optim
import subprocess
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_classification import train,val,test
from model.load_net import get_model 
from data.RNAGraph import RNADataset,collate_all
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import sys
import dgl
from sklearn.utils.class_weight import compute_class_weight
def plot(epochs,train_loss_list,val_loss_list,test_f1_score_list,save_path):
    epochs = list(range(epochs))
    plt.figure()
    plt.plot(epochs, train_loss_list, 'b*--', alpha=0.5, linewidth=1, label='train_loss')
    plt.plot(epochs, val_loss_list, 'rs--', alpha=0.5, linewidth=1, label='val_loss')
    plt.plot(epochs, test_f1_score_list, 'go--', alpha=0.5, linewidth=1, label='metric')
    plt.ylim(0.4, 1)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.savefig(f"{save_path}/loss_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

def select_free_gpu():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,nounits,noheader"],
            stderr=subprocess.STDOUT
        )
        # 解码并按行处理输出
        result = result.decode("utf-8").strip().splitlines()
        #print(result)
        #sys.exit()
        gpu_info = []
        for line in result:
            index, free_mem, total_mem = line.split(', ')
            free_mem = int(free_mem)
            total_mem = int(total_mem)
            gpu_info.append((index, free_mem, total_mem))
        print(gpu_info)
        gpu_info = sorted(gpu_info, key=lambda x: x[1], reverse=True)
        selected_gpu_id = gpu_info[0][0]
        free_memory = gpu_info[0][1]
        print(f"Selected GPU: {selected_gpu_id}, Free Memory: {free_memory} MiB")
        return selected_gpu_id

    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e.output.decode()}")
        return None  # 如果出错，返回 None，避免崩溃
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:' ,torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device
"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = get_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

def getmodel(MODEL_NAME,net_params):
    return get_model(MODEL_NAME, net_params)
#加载模型
def load_model(DATASET_NAME, MODEL_NAME, net_params, config, epoch):
    if config['debias'] == "True":
        save_dir = './model_save/debias/'
    else:
        save_dir = './model_save/bias/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir + DATASET_NAME) is False:
        os.makedirs(save_dir + DATASET_NAME)

    model = getmodel(MODEL_NAME, net_params)

    PATH = save_dir + DATASET_NAME + '/model_' + str(epoch) + '.pth'
    model.load_state_dict(torch.load(PATH))
    #model.eval()
    return model
#损失函数记录
def loss_record(DATASET_NAME, print_str, epoch, config,params):
    loss_dir = "./" + params["path"]+"/"
    if os.path.exists(loss_dir) is False:
        os.makedirs(loss_dir)
    if os.path.exists(loss_dir + DATASET_NAME + '.txt') is True and epoch == 0:
        os.remove(loss_dir + DATASET_NAME + '.txt')
    with open(loss_dir + DATASET_NAME + '.txt', 'a') as f:
        f.write(print_str + '\n')
#记录混淆举证
def matrix_recoder(dataset_name,params,roc_auc_score,confusion_matrix,epoch):
    basedir = "./" + params["path"]+"/confusion_matrix"
    if os.path.exists(basedir) is False:
        os.makedirs(basedir)
    with open(basedir+"/"+dataset_name+"_"+str(epoch)+".csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ROC AUC Score", roc_auc_score])
        writer.writerow([])
        writer.writerow(["Confusion Matrix"])
        # 写入混淆矩阵内容
        for row in confusion_matrix:
            writer.writerow(row)


    


#保存模型
def save_model(DATASET_NAME, model, epoch, config,params):
    # save
    m_path = params["path"]
    if config['debias'] == "True":
        save_dir = "./"+m_path+"/debias/"
    else:
        save_dir = "./"+m_path+"/bias/"
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir )
    torch.save(model.state_dict(), save_dir + '/model_' + str(epoch) + '.pth')
    #print("save_model:",save_dir)
def save_all_model(DATASET_NAME, model, epoch, config,params):
    # save
    m_path = params["path"]
    save_dir = "./"+m_path+"/all"
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    torch.save(model.state_dict(), save_dir  + '/model_' + str(epoch) + '.pth')
#训练
def train_all(MODEL_NAME, para, net_params, config, criterion, dirs =None):
    t0 = time.time()
    per_epoch_time = []
    best_val_auc = 0
    last_val_loss = 10
    best_epoch = 0
    early_stop_count = 0
    results_print_str = None
    print_list = []
    DATASET_NAME = para["dataset"]
    load_dataset_path = para["dataset_path"]
    base_dir = os.getcwd()

    trainset = RNADataset(load_dataset_path + "/train")
    valset =  RNADataset(load_dataset_path + "/test")
    testset =  RNADataset(load_dataset_path + "/test")
    train_loader = DataLoader(trainset, batch_size=para['batch_size'], shuffle=True, collate_fn=collate_all)
    test_loader = DataLoader(testset, batch_size=para['batch_size'], shuffle=False, collate_fn=collate_all)
    val_loader = DataLoader(valset, batch_size=para['batch_size'], shuffle=False, collate_fn=collate_all)


    device = net_params['device']
    print("model total param", net_params['total_param'])
    print("Number of Classes: ", net_params['n_classes'])
    # setting seeds
    random.seed(para['seed'])
    np.random.seed(para['seed'])
    torch.manual_seed(para['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(para['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    model = getmodel(MODEL_NAME, net_params)
    model = model.to(device)
    if not net_params['training']:
        for name, paras in model.RNALM.model.named_parameters():
            paras.requires_grad=False
    #打印完整模型参数
    print("model total paras:",sum(p.numel() for p in model.parameters() if p.requires_grad))
    #打印语言模型参数
    print('LM training paras', sum(p.numel() for p in model.RNALM.model.parameters() if p.requires_grad))

    
    optimizer = optim.Adam([
                   dict(params=[p for p in model.parameters() if p.requires_grad], lr=para['init_lr']),
                   dict(params=[p for p in model.RNALM.model.parameters() if p.requires_grad], lr=para['init_lr']/2),
                  ],
                  weight_decay=para['weight_decay'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=para['lr_reduce_factor'],
                                                     min_lr=para['min_lr'],
                                                     patience=para['lr_schedule_patience'],
                                                     verbose=True)
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []
    epoch_test_f1_score =[]



    try:
        with tqdm(range(para['epochs'])) as t:
             for epoch in t:
                 t.set_description('Epoch %d' % epoch)
                 start = time.time()
                 epoch_train_loss, epoch_train_acc, optimizer = train(model, optimizer,criterion,device, train_loader, epoch,use_fea=net_params['fea_info'],use_softmax =net_params["use_softmax"])
                 epoch_val_loss, epoch_val_acc = val(model,criterion,device, val_loader, epoch,use_fea=net_params['fea_info'],use_softmax =net_params["use_softmax"])
                 epoch_test_acc,\
                 epoch_test_auc,\
                 epoch_test_f1,\
                 epoch_test_precision,\
                 epoch_test_recall,\
                 epoch_test_confusion_matrix= test(model,device, test_loader, epoch,para["path"],use_fea=net_params['fea_info'],use_softmax =net_params["use_softmax"])

                 epoch_train_losses.append(epoch_train_loss)
                 epoch_val_losses.append(epoch_val_loss)
                 epoch_train_accs.append(epoch_train_acc)
                 epoch_val_accs.append(epoch_val_acc)
                 epoch_test_f1_score.append(epoch_test_f1)

                 print_str = 'epoch ' + str(epoch) + \
                             ' train_loss ' + str(epoch_train_loss) + ' train_acc ' + str(epoch_train_acc) + \
                             ' val_loss ' + str(epoch_val_loss) + ' val_acc ' + str(epoch_val_acc) + \
                             ' test_acc ' + str(epoch_test_acc) + \
                             ' test_roc_auc ' + str(epoch_test_auc) + ' test_f1 '+str(epoch_test_f1)+\
                             ' test_precision ' + str(epoch_test_precision) + ' test_recall ' + str(epoch_test_recall)
                 print_list.append(print_str)
                 per_epoch_time.append(time.time( ) -start)

                 scheduler.step(epoch_val_loss)
                 #scheduler.step()
                 #记录所有信息
                 loss_record(DATASET_NAME, print_list[-1], epoch, config,para)
                 #记录roc_auc_score, confusion_matrix
                 matrix_recoder(DATASET_NAME,para,epoch_test_auc,epoch_test_confusion_matrix,epoch)
                 #保存所有模型
                 save_all_model(DATASET_NAME, model, epoch, config,para)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    ###绘制深度学习图表
    plot(para['epochs'],epoch_train_losses,epoch_val_losses,epoch_test_f1_score,"./" + para["path"])




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_name',help='Please give a save path')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()
    gpu_id = select_free_gpu()
    if gpu_id is not None:
        print(f"Selected GPU: {gpu_id}")
    else:
         #默认使用第六张卡
         gpu_id = 6
         print("No suitable GPU found.")  
    gpu_use = True
    device = gpu_setup(gpu_use,gpu_id)
    base_conf = {
    "gpu": {
        "use": True,
        "id": 0
    },
    "dataset": "MNIST",
    "debias": "False",
    "motif": "False",
    "best_epoch": 0, 
}
    MODEL_NAME = "IreeSeek_LM_model"

#######参数处理##########
    para ={
        "seed": 39,
        "epochs": args.epoch,#300 200 100
        "batch_size": args.batch_size, #128
        "init_lr": args.lr,# 0.002 0.001 0.0005
        "lr_reduce_factor": 0.8,#lr*=factor, origin=0.5
        "lr_schedule_patience": args.epoch//10,#10  50
        "min_lr": 5e-5, # 1e-5
        "weight_decay": 0.01,
        "print_epoch_interval": 5,
        "max_time": 24,
        "dataset": "ires",
    }
    para["path"] = args.run_name
    
    
    is_oversamplng = False
    if is_oversamplng:
        labels = np.array([0] * (9034) + [1] * (11847))
        #labels = np.array([0] * 9034 + [1] * 19745)
        para["dataset_path"] = os.getcwd() +"/RNAFinalData/oversampling"
    else:
        labels =  np.array([0] * 3949 + [1] * 3949)
        para["dataset_path"] = os.getcwd() +"/RNAFinalData/undersampling/ensemble_3"
    print("args:",para["path"],"dataset_path:",para["dataset_path"])
    
    
    
    net_params = {
        "L": 2,
        "in_dim":4,
        "hidden_dim": 32,
        "out_dim": 32,
        "residual": False,
        "readout": "mean",
        "in_feat_dropout": 0.1,
        "dropout": 0.1,
        "batch_norm": True,
        "self_loop": False,
        "embedding_dim":0,
        "gpu_id": gpu_id,
        "type": "GCN",
        "n_classes": 2,
        "use_softmax":True
    }
    net_params['device'] = device
    net_params['fea_info'] = 4 

    #语言模型参数设置
    net_params['pretrain'] =  os.getenv('structRFM_checkpoint')
    net_params['training'] = True


    print("params:",para)
    print("net_params:",net_params)
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    # 计算类别权重
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(net_params['device'])
    print("class_weights:",class_weights)
    
    # 传入CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #criterion = nn.CrossEntropyLoss()


    train_all(MODEL_NAME, para, net_params, base_conf,criterion)





main()
    
