import argparse
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,Dataset
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# from pytorchtools import EarlyStopping
from modeltask2 import FusionLayer1, Model1, Model2, Model4, BertMLP
from torch_geometric.nn import GAE, VGAE
from transformers import BertTokenizer, BertModel
# from bert import BertForMultiLable, FusionLayer
from collections import defaultdict
import os
import random
from tqdm import tqdm

# from long_code import *
import torch
from transformers import AutoTokenizer, BloomModel

from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
# from bert import BertMLM, FusionLayer
import math
import gc
from copy import deepcopy
import json
from itertools import combinations

path_prefix = r'/home/jy/桌面/MSE-ENN/'

# baike_duiqi = pd.read_excel(path_prefix + '/drugdata/baike_duiqi_original.xlsx', sheet_name="baike_duiqi_info_zh")


# 'cuda' if torch.cuda.is_available() else 
device = torch.device('cuda:0')

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches",type=int,default=90)
parser.add_argument("--batch_size",type=int,default=1024)
parser.add_argument("--weigh_decay",type=float,default=1e-8)
parser.add_argument("--lr",type=float,default=5*1e-4) #5*1e-4 1e-4
# 对于不同的数据可以尝试不同采样数量
parser.add_argument("--neighbor_sample_size",type=int,default=6)
parser.add_argument("--event_num",type=int,default=4)

parser.add_argument("--n_drug",type=int,default=1409)
parser.add_argument("--seed",type=int,default=1)
parser.add_argument("--dropout",type=float,default=0.5)
parser.add_argument("--embedding_num",type=int,default=384)
parser.add_argument("--input_num",type=int,default=512)
parser.add_argument("--sample",type=int,default=1)
args = parser.parse_args()

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def read_dataset(drug_name_id,num, values, values_num):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list=[]
    filename = "/home/jy/桌面/MSE-ENN/dataset5/dataset" +str(num)+ ".txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string= line.rstrip().split(',',2)
            head=string[0]
            tail=string[1]
            relation=string[2]

            drug_list.append(drug_name_id[head])
            
            if tail not in tails:
                tails[tail] = len(tails)
                if tail in values:
                    values_num.append(tails[tail])
            if relation not in relations:
                relations[relation] = len(relations)
            if num==3:
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    
    return kg,len(tails),len(relations), tails


def prepare(mechanism_action):
    d_label = {}
    d_event = []
    new_label = []
    for i in range(len(mechanism_action)):
        d_event.append(mechanism_action[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        # if 'metabolism' in list1[i][0] or 'concentration' in list1[i][0]:
        #     print(i, list1[i][0])
        d_label[list1[i][0]] = i
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])
    return new_label,len(count)


def l2_re(parameter):
    reg=0
    for param in parameter:
        reg+=0.5*(param**2).sum()
    return reg

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    # label_binarize:返回一个one_hot的类型
    y_one_hot = label_binarize(y_test, classes = np.arange(event_num))
    pred_one_hot = label_binarize(pred_type,classes = np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = 0.0
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = 0.0
    result_all[5] = 0.0
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = 0.0
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = 0.0
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = 0.0
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]

def save_result(filepath,result_type,result):
    with open(filepath+result_type +'task3'+ '.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def is_nan(s):
    try:
        f = float(s)
        return math.isnan(f)
    except ValueError:
        return False


def train_gpu(train_x,train_y,test_x,test_y, dataset_tail1, dataset_tail2, dataset_tail4, net,test_adj, dict1, tokenizer, drug_emb, real_edge_1, real_edge_2, real_edge_4, feats_1, feats_2, feats_4, main_l, aux_l, aux_l2, dataset_kgs=None, substructure_tails=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:0')
    net.to(device)
    
    # loss_i = SoftWeightedLoss(num_tasks=4)

    loss_function=nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weigh_decay)
    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    
    train_x_total = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)

    train_data = TensorDataset(train_x_total, train_y)

    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    test_list = []
    max_test_output = torch.zeros((0,4),dtype=torch.float).to(device)
    real_edge_1 = real_edge_1.to('cuda:0')
    real_edge_2 = real_edge_2.to('cuda:0')
    real_edge_4 = real_edge_4.to('cuda:0')
    

    bert_feats = torch.load('/home/jy/桌面/MSE-ENN/dataset5/drug_emb.pt').cpu().numpy()
    feats = {'feats_1': feats_1, 'feats_2': feats_2, 'feats_4': feats_4, 'feats_bert': bert_feats}
    
    for epoch in range(args.epoches):
        test_loss, test_score, train_l = 0, 0, 0
        train_a = []
        net.train()
        # loss_i.train()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)

            opti.zero_grad()
            train_acc = 0
            train_label = y

            final_embed, z4, z2, z1, emb4, emb2, emb1 = net((x, feats))

            main_loss = loss_function(final_embed, train_label)
            recon1 = loss_function(z1, real_edge_1)
            recon2 = loss_function(z2, real_edge_2)
            recon3 = loss_function(z4, real_edge_4)

            
            l = 1 * main_loss + \
                0.06 * recon1 + \
                0.06 * recon2 + \
                0.06 * recon3 
            
            l.backward()
            
            opti.step()

            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(final_embed, dim=1).cpu().numpy(), train_label.cpu().numpy())
            train_a.append(train_acc)
        net.eval()
        # loss_i.eval()
        with torch.inference_mode():
            test_x = torch.LongTensor(test_x)
            test_x_gpu = torch.LongTensor(test_x).to(device)

            # f_input = [test_x_gpu, torch.tensor(1).to(device), drug_emb.to(device)]

            test_final_embed, test_emb4, test_emb2, test_emb1, test_z4, test_z2, test_z1 = net((test_x_gpu, feats))
            test_output = F.softmax(test_final_embed, dim=1)
            test_label = torch.LongTensor(test_y).to(device)
            loss = 1 * loss_function(test_output, test_label) + \
                    0.06 * loss_function(test_emb4, real_edge_4) + \
                    0.06 * loss_function(test_emb2, real_edge_2) + \
                    0.06 * loss_function(test_emb1, real_edge_1)
            
            # loss = loss_i([loss_function(final_embed, train_label), loss_function(test_z1, real_edge_1), loss_function(test_z2, real_edge_2), loss_function(test_z4, real_edge_4)], 1)
            # print('主：', loss_function(final_embed, train_label), '辅：', loss_i([loss_function(test_z1, real_edge_1), loss_function(test_z2, real_edge_2), loss_function(test_z4, real_edge_4)]))
            # loss = loss_function(test_output, test_label) + loss_function(test_z1, real_edge_1) + loss_function(test_z2, real_edge_2) + loss_function(test_z4, real_edge_4)
            test_loss = loss.item()
            test_output = test_output.to('cpu')
            test_score = f1_score(torch.argmax(test_output, dim=1).cpu().numpy(), test_label.cpu().numpy(), average='macro')
            test_acc = accuracy_score(torch.argmax(test_output, dim=1).cpu().numpy(), test_label.cpu().numpy())
            test_list.append(test_acc)
            if test_acc == max(test_list):
                max_test_output = test_output
            print("test_acc:", test_acc, "train_acc:", sum(train_a) / len(train_a),"test_score:",test_score)
        print('epoch [%d] train_loss: %.6f testing_loss: %.6f ' % (
                epoch + 1, train_l / len(train_y), test_loss / len(test_y)))
        
    return test_loss / len(test_y), max(test_list), train_l / len(train_y), sum(train_a) / len(
        train_a), test_list, max_test_output

def main():
    ddi_file_path = '/home/jy/桌面/MSE-ENN/dataset5/DDI1409DB.csv'
    drug_file_path = '/home/jy/桌面/MSE-ENN/dataset5/smi_DB.csv'
    df_drug = pd.read_csv(drug_file_path)
    extraction = pd.read_csv(ddi_file_path)

    mechanism = extraction['DDI']
    drugA = extraction['drug_A']
    drugB = extraction['drug_B']
    # new_label,event_num = prepare(mechanism_action)
    new_label,event_num = prepare(mechanism)
    new_label = np.array(new_label)
    dict1 = {}
    for i in df_drug["DB_ID"]:
        dict1[i] = len(dict1)

    drug_name = [dict1[i] for i in df_drug["DB_ID"]]
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]

    maccs = []

    dataset1_kg, dataset1_tail_len, dataset1_relation_len, dataset1_tails = read_dataset(dict1,1,entities, [])
    dataset2_kg, dataset2_tail_len, dataset2_relation_len, dataset2_tails = read_dataset(dict1,2, substructures, [])
    # dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1,3)
    dataset4_kg, dataset4_tail_len, dataset4_relation_len, dataset4_tails = read_dataset(dict1,4, maccs, [])
    x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
    substructures_dict = {i: dataset2_tails[i] for i in substructures}
    x_datasets = pd.DataFrame(data=x_datasets)
    x_datasets = x_datasets.to_numpy()
    dataset={}
    dataset["dataset1"],dataset["dataset2"],dataset["dataset4"] = dataset1_kg,dataset2_kg,dataset4_kg
    tail_len={}
    tail_len["dataset1"],tail_len["dataset2"],tail_len["dataset4"] = dataset1_tail_len,dataset2_tail_len,dataset4_tail_len
    relation_len={}
    relation_len["dataset1"],relation_len["dataset2"],relation_len["dataset4"] = dataset1_relation_len,dataset2_relation_len,dataset4_relation_len
    train_sum, test_sum = 0, 0

    feature_matrix_origin = torch.randn((1409+825, 256), dtype=float)
    # 药物与药物子结构的节点与特征
    feats = {}

    feats_1 = np.zeros((1409, dataset1_tail_len), dtype=float)
    feats_2 = np.zeros((1409, dataset2_tail_len), dtype=float)
    feats_4 = np.zeros((1409, dataset4_tail_len), dtype=float)

    # 构建edge_index
    edge_index_1 = [[], []]
    edge_index_2 = [[], []]
    edge_index_4 = [[], []]

    temp = {}
    real_edge_1 = []
    real_edge_2 = []
    real_edge_4 = []
    num_classes_1 = []
    num_classes_2 = []
    num_classes_4 = []

    temp_kg = [defaultdict(list) for i in range(4)]

    for p,kg in enumerate(dataset):
        for i in dataset[kg].keys():
            for j in dataset[kg][i]:
                temp_kg[p][i].append(j[0])
                # print(j)
                if p == 0:
                    if i not in temp:
                        temp[i] = [j]
                    else:
                        temp[i].append(j)
                    feats_1[i][j[0]] = j[1]
                    edge_index_1[0].append(i)
                    edge_index_1[1].append(1409 + j[0])
                    real_edge_1.append(j[1])
                    num_classes_1.append(j[1])
                elif p == 1:
                    feats_2[i][j[0]] = 1
                    edge_index_2[0].append(i)
                    edge_index_2[1].append(1409 + j[0])
                    real_edge_2.append(1)
                elif p == 2:
                    feats_4[i][j[0]] = 1  # 修复：dataset4的relation只有include，索引为0，所以用1表示存在
                    edge_index_4[0].append(i)
                    edge_index_4[1].append(1409 + j[0])
                    real_edge_4.append(1)
                    num_classes_4.append(1)

    edge_index_1 = torch.tensor(edge_index_1)
    edge_index_2 = torch.tensor(edge_index_2)
    edge_index_4 = torch.tensor(edge_index_4)

    num_classes_1 = set(num_classes_1)
    num_classes_4 = set(num_classes_4)

    # from torch_geometric.data import HeteroData
    # data = HeteroData()

    # data['drug'].x = torch.randn(1409, 256)
    # data['chemical'].x = torch.randn(825, 256)


    feature_matrix1 = np.zeros((1409, dataset1_tail_len), dtype=float)
    feature_matrix2 = np.zeros((1409, dataset2_tail_len), dtype=float)
    feature_matrix3 = np.zeros((1409, dataset4_tail_len), dtype=float)
    feature_matrix4 = np.zeros((1409, 1409), dtype=float)

    for i in dataset4_kg.keys():
        for p,v in dataset4_kg[i]:
            feature_matrix3[i][p] = v

    for i in temp_kg[0].keys():
        for j in temp_kg[0][i]:
            feature_matrix1[i][j] = 1

    for i in temp_kg[1].keys():
        for j in temp_kg[1][i]:
            feature_matrix2[i][j] = 1

    for i in temp_kg[2].keys():
        for j in temp_kg[2][i]:
            feature_matrix4[i][j] = 1

    temp_drugA = [[] for i in range(event_num)]
    temp_drugB = [[] for i in range(event_num)]
    for i in range(len(new_label)):
        temp_drugA[new_label[i]].append(drugA_id[i])
        temp_drugB[new_label[i]].append(drugB_id[i])

    drug_cro_dict = {}
    for i in range(event_num):
        for j in range(len(temp_drugA[i])):
            drug_cro_dict[temp_drugA[i][j]] = j % 5
            drug_cro_dict[temp_drugB[i][j]] = j % 5

    train_drug = [[] for i in range(5)]
    test_drug = [[] for i in range(5)]
    # 将全部数据分为test data 和 train data,
    for i in range(5):
        for dr_key in drug_cro_dict.keys():
            if drug_cro_dict[dr_key] == i:
                test_drug[i].append(dr_key)
            else:
                train_drug[i].append(dr_key)

    test_adj = [[defaultdict(list) for _ in range(4)] for _ in range(5)]
    
    y_true = np.array([])
    y_score = np.zeros((0, 4), dtype=float)
    y_pred = np.array([])
    tokenizer = ''

    real_edge_1 = torch.tensor(real_edge_1)
    real_edge_2 = torch.tensor(real_edge_2)
    real_edge_4 = torch.tensor(real_edge_4)
    drug_emb = torch.load('/home/jy/桌面/MSE-ENN/dataset5/drug_emb.pt')

    for cross_ver in range(0, 5):

            net = nn.Sequential(Model1(args, dataset1_tail_len, feats_1, edge_index_1, 512, args.embedding_num, args.embedding_num, args.embedding_num, len(num_classes_1), real_edge_1),
                                Model2(args, dataset2_tail_len, feats_2, edge_index_2, 512, args.embedding_num, args.embedding_num, args.embedding_num, 2, real_edge_2),
                                Model4(args, dataset4_tail_len, feats_4, edge_index_4, 512, args.embedding_num, args.embedding_num, args.embedding_num, 2, real_edge_4),
                                BertMLP('..'),
                                FusionLayer1(args))
            # net = torch.load('/home/jy/桌面/MSE-ENN/interpret_kernel_1409/1drug_id_str/drug_id_str_modality.pth', map_location=device)
            
            X_train = []
            X_test = []
            y_train = []
            y_test = []
            for i in range(len(drugA)):
                if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                    X_train.append(i)
                    y_train.append(i)

                if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] not in np.array(train_drug[cross_ver])):
                    X_test.append(i)
                    y_test.append(i)
                
                if (drugA_id[i] not in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                    X_test.append(i)
                    y_test.append(i)
            
            train_x = x_datasets[X_train]
            train_y = new_label[y_train]
            test_x = x_datasets[X_test]
            test_y = new_label[y_test]

            test_loss, test_acc, train_loss, train_acc, test_list, test_output = train_gpu(train_x, train_y, test_x, test_y, dataset1_tails, dataset2_tails, dataset4_tails,
                                                                            net,test_adj[0], dict1, tokenizer, drug_emb, real_edge_1, real_edge_2, real_edge_4, feats_1, feats_2, feats_4, 2, 2, 2, dataset_kgs=dataset)
    
    # torch.save(net, '/home/jy/桌面/MSE-ENN/interpret_kernel_1409/1drug_id_str/drug_id_str_modality.pth')
    train_sum += train_acc
    test_sum += test_acc
    test_output = test_output.to('cpu')
    pred_type = torch.argmax(test_output, dim=1).numpy()
    y_pred = np.hstack((y_pred, pred_type))
    y_score = np.row_stack((y_score, test_output))
    y_true = np.hstack((y_true, test_y))
    print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
            0, test_loss, test_acc, train_loss, train_acc))

    result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
    save_result(path_prefix + "/result_dataset5_task2/loss_mlp_" + str(1) + '_' + str(2) + '_' + str(3)  + '_', "all", result_all)
    # save_result(path_prefix + "/new_result2/loss_mlp_" + str(main_l) + '_' + str(l) + '_', "each", result_eve)
    print('%d-fold validation: avg train acc  %f, avg test acc %f' % (5, train_sum / 5, test_sum / 5))
    return

if __name__ == '__main__':
    main()
