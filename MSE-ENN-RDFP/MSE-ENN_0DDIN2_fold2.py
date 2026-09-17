import argparse
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
from modeltask2 import FusionLayer1, Model1, Model2, Model4, BertMLP, ModalityCrossAttentionFusion
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
import warnings
# from bert import BertMLM, FusionLayer
import math
import gc

path_prefix = r'/Users/charly/Downloads'

# baike_duiqi = pd.read_excel(path_prefix + '/drugdata/baike_duiqi_original.xlsx', sheet_name="baike_duiqi_info_zh")


# 'cuda' if torch.cuda.is_available() else 
device = torch.device('cuda:1')

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches",type=int,choices=[100,500,1000,2000],default=90)
parser.add_argument("--batch_size",type=int,choices=[2048,1024,512,256,128],default=1024)
parser.add_argument("--weigh_decay",type=float,choices=[1e-1,1e-2,1e-3,1e-4,1e-8],default=1e-8)
parser.add_argument("--lr",type=float,choices=[1e-3,1e-4,1e-5,6*1e-3],default=4*1e-4) #5*1e-4 1e-4
# 对于不同的数据可以尝试不同采样数量
parser.add_argument("--neighbor_sample_size",choices=[4,6,10,16],type=int,default=6)
parser.add_argument("--event_num",type=int,default=86)

parser.add_argument("--n_drug",type=int,default=1710)
parser.add_argument("--seed",type=int,default=1)
parser.add_argument("--dropout",type=float,default=0.5)
parser.add_argument("--embedding_num",type=int,choices=[128,64,256,32],default=384)
parser.add_argument("--sample",type=int,choices=[1,2,3,4,5,6],default=1)
parser.add_argument("--dataset_dir", type=str, default="/home/jy/桌面/MKG/dataset_mecddi")
parser.add_argument("--ddi_file", type=str, default="final.csv")
parser.add_argument("--drug_file", type=str, default="drug_smiles.csv")
parser.add_argument("--dataset1_file", type=str, default="dataset1.txt")
parser.add_argument("--dataset2_file", type=str, default="dataset2.txt")
parser.add_argument("--dataset4_file", type=str, default="dataset4.txt")
parser.add_argument("--drug_emb_path", type=str, default="/home/jy/桌面/MKG/dataset3/drug_emb_dataset3.pt")
parser.add_argument("--result_dir", type=str, default="/home/jy/桌面/MKG/MSREN-1710-rdfp/outputs/results")
parser.add_argument("--run_rdfp", type=int, choices=[0, 1], default=1)
parser.add_argument("--rdfp_fold", type=int, choices=[0, 1, 2, 3, 4], default=2)
parser.add_argument("--rdfp_top_k", type=int, default=3)
parser.add_argument("--rdfp_output_dir", type=str, default="/home/jy/桌面/MKG/MSREN-1710-rdfp/outputs/rdfp_fold2")
args = parser.parse_args()

# seed = 0

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


def read_dataset(drug_name_id, file_path):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list=[]
    with open(file_path, encoding="utf8") as reader:
        for line in reader:
            string= line.rstrip().split('//',2)
            head=string[0]
            tail=string[1]
            relation=string[2]
            drug_list.append(drug_name_id[head])
            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)
            if "dataset3" in os.path.basename(file_path):
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    return kg,len(tails),len(relations),tails

# def prepare(mechanism, action):
#     d_label = {}
#     d_event = []
#     new_label = []
#     for i in range(len(mechanism)):
#         d_event.append(mechanism[i] + " " + action[i])
#     count = {}
#     for i in d_event:
#         if i in count:
#             count[i] += 1
#         else:
#             count[i] = 1
#     list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
#     for i in range(len(list1)):
#         d_label[list1[i][0]] = i
#     for i in range(len(d_event)):
#         new_label.append(d_label[d_event[i]])
#     return new_label,len(count)


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
        result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average=None)
        # result_eve[i, 2] = 0.0
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

def get_desc_list(x, dict1, baike_duiqi):
    descs1 = []
    descs2 = []
    nullList = []
    baike_duiqi['en_name'] = baike_duiqi['en_name'].str.lower().str.replace(' ', '')
    # baike_duiqi = baike_duiqi.dropna()
    # baike_duiqi['all'] = baike_duiqi['description'] + baike_duiqi['indication'] + baike_duiqi['categories'] + baike_duiqi['adverse_reaction'] + baike_duiqi['drug_interaction'] + baike_duiqi['pharmacology_effect']
    for li in x:
        descs = ''
        # 分开存
        for num, id in enumerate(li):
            for key, val in dict1.items():
                if val == id:
                    indices_with_id = baike_duiqi['en_name'][baike_duiqi['en_name'].str.contains(key.lower())].index.values
                    filtered_df = baike_duiqi.loc[baike_duiqi.index.isin(indices_with_id)]
                    desc = ''
                    if len(indices_with_id) == 0:
                        desc = ''
                        nullList.append(key)
                    else:
                        desc = filtered_df.iloc[0, :]['description']
                    if num == 0:
                        descs1.append(desc)
                    else:
                        descs2.append(desc)
    return descs1 + descs2

def is_nan(s):
    try:
        f = float(s)
        return math.isnan(f)
    except ValueError:
        return False

def get_desc_list2(dict1):
    descs = []
    max_size = 0
    baike_duiqi['en_name'] = baike_duiqi['en_name'].str.lower().str.replace(' ', '')
    for drug, id in dict1.items():
        drug = drug.replace(' ', '')
        indices_with_id = baike_duiqi['en_name'][baike_duiqi['en_name'].str.contains(drug.lower())].index.values
        filtered_df = baike_duiqi.loc[baike_duiqi.index.isin(indices_with_id)]
        if len(indices_with_id) == 0:
            print(drug)
            descs.append('暂无描述')
        else:
            obj = filtered_df.iloc[0, :]                                
            
            if is_nan(obj['description']):
                if len(indices_with_id) > 1:
                    obj = filtered_df.iloc[1, :]
            if is_nan(obj['categories']):
                if len(indices_with_id) > 1:
                    obj = filtered_df.iloc[1, :] 
            if len(obj['description']) > max_size:
                max_size = len(obj['description'])
            # 添加提示词
            descs.append(obj['zh_name'] + ',' + obj['en_name'] + '的药品描述为: ' + obj['description'])
    print(max_size)
    return descs

def get_desc_list3(x, dict1):
    desc_list = []
    baike_duiqi['en_name'] = baike_duiqi['en_name'].str.lower().str.replace(' ', '')
    for li in x:
        descs = ''
        for num, id in enumerate(li):
            for key, val in dict1.items():
                if val == id:
                    indices_with_id = baike_duiqi['en_name'][baike_duiqi['en_name'].str.contains(key.lower())].index.values
                    filtered_df = baike_duiqi.loc[baike_duiqi.index.isin(indices_with_id)]
                    desc = ''
                    if len(indices_with_id) == 0:
                        desc = ''
                    else:
                        desc = filtered_df.iloc[0, :]['description']
                    descs += desc
        desc_list.append(descs)
    return desc_list

# tokenizer = AutoTokenizer.from_pretrained('/home/jy/桌面/MKG/model/udever')
# model = BloomModel.from_pretrained('/home/jy/桌面/MKG/model/udever')

# boq, eoq, bod, eod = '[BOQ]', '[EOQ]', '[BOD]', '[EOD]'
# eoq_id, eod_id = tokenizer.convert_tokens_to_ids([eoq, eod])
def encode(texts: list, is_query: bool = True, max_length=203):
    bos = boq if is_query else bod
    eos_id = eoq_id if is_query else eod_id
    all_texts = torch.tensor([])
    for text in tqdm(texts):
        text = [bos + text]
        encoding = tokenizer(
            text, truncation=True, max_length=max_length - 1, padding=True
        )
        for ids, mask in zip(encoding['input_ids'], encoding['attention_mask']):
            ids.append(eos_id)
            mask.append(1)
        inputs = tokenizer.pad(encoding, return_tensors='pt')
        inputs = {k:v for k,v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            embeds = outputs.last_hidden_state[:, -1]
        all_texts = torch.cat([all_texts, embeds], 0)
    
    # print(all_texts.shape)
    return torch.tensor(all_texts)


def train_gpu(train_x,train_y,test_x,test_y,net,test_adj, dict1, tokenizer, drug_emb, real_edge_1, real_edge_2, real_edge_4, feats_1, feats_2, feats_4, main_l, aux_l, aux_l2):

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cuda:1')
    net.to(device)
    
    # loss_i = SoftWeightedLoss(num_tasks=4)

    loss_function=nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weigh_decay)
    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    train_x1 = train_x.copy()
    train_x[:,[0,1]] = train_x[:,[1,0]]
    
    train_x_total = torch.LongTensor(np.concatenate([train_x1, train_x], axis=0))
    train_y = torch.LongTensor(np.concatenate([train_y,train_y]))

    train_data = TensorDataset(train_x_total, train_y)

    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    test_list = []
    max_test_output = torch.zeros((0,86),dtype=torch.float).to(device)
    real_edge_1 = real_edge_1.to('cuda:1')
    real_edge_2 = real_edge_2.to('cuda:1')
    real_edge_4 = real_edge_4.to('cuda:1')
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
            # final = net()

            # x = torch.LongTensor(x)
            
            # f_input = [x, torch.tensor(0).to(device), defaultdict(list), drug_emb.to(device)]
            # f_input = [x, ]
            final_embed, emb1, emb2, emb4, z4, z2, z1 = net(x)

            # print(loss_function(final_embed, train_label), loss_function(z1, real_edge_1), loss_function(z2, real_edge_2), loss_function(z4, real_edge_4))
            
            main_loss = loss_function(final_embed, train_label)
            recon1 = loss_function(z1, real_edge_1)
            recon2 = loss_function(z2, real_edge_2)
            recon3 = loss_function(z4, real_edge_4)
            gae1_loss = loss_function(emb1, train_label)

            
            l = main_l * main_loss + \
                aux_l * recon1 + \
                aux_l * recon2 + \
                aux_l * recon3
            
            l.backward()
            
            opti.step()

            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(final_embed, dim=1).cpu().numpy(), train_label.cpu().numpy())
            train_a.append(train_acc)
        net.eval()
        # loss_i.eval()
        with torch.no_grad():
            test_x = torch.LongTensor(test_x)
            test_x_gpu = torch.LongTensor(test_x).to(device)

            # f_input = [test_x_gpu, torch.tensor(1).to(device), drug_emb.to(device)]

            test_final_embed, test_emb1, test_emb2, test_emb3, test_z4, test_z2, test_z1 = net(test_x_gpu)
            test_output = F.softmax(test_final_embed, dim=1)
            test_label = torch.LongTensor(test_y).to(device)
            loss = main_l * loss_function(test_output, test_label) + \
                    aux_l * loss_function(test_z1, real_edge_1) + \
                    aux_l * loss_function(test_z2, real_edge_2) + \
                    aux_l * loss_function(test_emb1, test_label)
            # 0.000000001
            
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

def train(train_x,train_y,test_x,test_y,net,test_adj, dict1, tokenizer, drug_emb):
    loss_function=nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weigh_decay)
    # opti2 = torch.optim.Adam(bert_model.parameters(), lr=args.lr,weight_decay=args.weigh_decay)
    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    train_x1 = train_x.copy()
    train_x[:,[0,1]] = train_x[:,[1,0]]
    
    train_x_total = torch.LongTensor(np.concatenate([train_x1, train_x], axis=0))
    train_y = torch.LongTensor(np.concatenate([train_y,train_y]))

    train_data = TensorDataset(train_x_total, train_y)

    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    test_list = []
    max_test_output = torch.zeros((0,86),dtype=torch.float)
    for epoch in range(args.epoches):
        test_loss, test_score, train_l = 0, 0, 0
        train_a = []
        net[0].train()
        net[1].train()
        net[2].train()
        for x, y in tqdm(train_iter):
            opti.zero_grad()
            # opti2.zero_grad()
            train_acc = 0
            train_label = torch.LongTensor(y)
            x = torch.LongTensor(x)
            x_copy = x[:]
            f_input = list()
            f_input.append(x)
            f_input.append(0)
            f_input.append(defaultdict(list))

            arg2 = {
                'x': x,
                'f_input': f_input,
                'emb': drug_emb
            }
            net_output = net(arg2)
            l = loss_function(net_output, train_label)
            l.backward()
            opti.step()

            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(net_output,dim=1), train_label)
            train_a.append(train_acc)
        net[0].eval()
        net[1].eval()
        net[2].eval()
        with torch.no_grad():
            test_x = torch.LongTensor(test_x)
            test_x_copy = test_x[:]
            f_input = list()
            f_input.append(test_x)
            f_input.append(1)
            f_input.append(test_adj)
            arg2 = {
                'x': test_x,
                'f_input': f_input,
                'emb': drug_emb
            }
            test_output = F.softmax(net(arg2),dim=1)
            test_label = torch.LongTensor(test_y)
            loss = loss_function(test_output, test_label)
            test_loss = loss.item()
            test_score = f1_score(torch.argmax(test_output,dim=1), test_label,average='macro')
            test_acc = accuracy_score(torch.argmax(test_output,dim=1), test_label)
            test_list.append(test_score)
            if test_score == max(test_list):
                max_test_output = test_output
            print("test_acc:", test_acc, "train_acc:", sum(train_a) / len(train_a),"test_score:",test_score)
        print('epoch [%d] train_loss: %.6f testing_loss: %.6f ' % (
                epoch + 1, train_l / len(train_y), test_loss / len(test_y)))
    return test_loss / len(test_y), max(test_list), train_l / len(train_y), sum(train_a) / len(
        train_a), test_list, max_test_output

def find_dif(raw_matrix):
    n_drug = raw_matrix.shape[0]
    n_feat = raw_matrix.shape[1]
    sim_matrix4 = np.zeros((n_drug, n_drug), dtype=float)
    for i in range(n_drug):
        for j in range(n_drug):
            for k in range(n_feat):
                if i==j:
                    sim_matrix4[i,j] = 0
                    break
                else:
                    if raw_matrix[i,k] == raw_matrix[j,k]:
                        sim_matrix4[i, j] += 1
                    else:
                        sim_matrix4[i, j] += 0
    return sim_matrix4

def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return numerator / denominator

def invert_index_mapping(name_to_index):
    return {idx: name for name, idx in name_to_index.items()}

def resolve_feature_name(modality_name, feat_idx, feature_name_maps):
    if modality_name == "text_description" and feat_idx == -1:
        return "text_description[ALL]"
    raw_name = feature_name_maps.get(modality_name, {}).get(int(feat_idx), str(feat_idx))
    return str(raw_name)

def get_pair_fused_embedding(net, pair_tensor):
    with torch.no_grad():
        out = net[0](pair_tensor)
        out = net[1](out)
        out = net[2](out)
        out = net[3](out)
        llm_embedding, gae4_embedding, _, gae2_embedding, _, gae1_embedding, _, idx = out
        drugA = idx[:, 0]
        drugB = idx[:, 1]
        fused_embedding = torch.cat(
            [
                gae1_embedding[drugA], gae2_embedding[drugA], gae4_embedding[drugA], llm_embedding[drugA],
                gae1_embedding[drugB], gae2_embedding[drugB], gae4_embedding[drugB], llm_embedding[drugB]
            ],
            dim=1
        ).float()
    return fused_embedding

def build_pair_feature_specs(net, drug_a_id, drug_b_id):
    # 按“药物对（A+B）”组织输入特征：包含A/B各自全部非零输入位点，
    # 同名位点合并为一个pair特征，并记录应mask A/B哪一侧。
    modality_tensors = [
        ("chemical_entities", net[0].autoencoder.drug_features),
        ("morgan_fingerprint", net[1].autoencoder.drug_features),
        ("maccs_fingerprint", net[2].autoencoder.drug_features),
    ]
    feature_specs = []
    for modality_name, feature_tensor in modality_tensors:
        a_index = set(torch.nonzero(feature_tensor[drug_a_id] != 0, as_tuple=False).view(-1).tolist())
        b_index = set(torch.nonzero(feature_tensor[drug_b_id] != 0, as_tuple=False).view(-1).tolist())
        pair_index = sorted(a_index | b_index)
        for feat_idx in pair_index:
            mask_a = feat_idx in a_index
            mask_b = feat_idx in b_index
            feature_specs.append(
                {
                    "modality": modality_name,
                    "feature_index": int(feat_idx),
                    "mask_a": mask_a,
                    "mask_b": mask_b,
                    "source": "A&B" if (mask_a and mask_b) else ("A" if mask_a else "B"),
                }
            )
    return feature_specs

def masked_feature_rdfp_score(net, pair_tensor, base_embedding, drug_a_id, drug_b_id, modality_name, feat_idx, mask_a, mask_b):
    modality_tensor_map = {
        "chemical_entities": net[0].autoencoder.drug_features,
        "morgan_fingerprint": net[1].autoencoder.drug_features,
        "maccs_fingerprint": net[2].autoencoder.drug_features,
        "text_description": net[3].drug_emb,
    }
    target_tensor = modality_tensor_map[modality_name]
    if modality_name == "text_description" and feat_idx == -1:
        old_a = target_tensor[drug_a_id].clone() if mask_a else None
        old_b = target_tensor[drug_b_id].clone() if mask_b else None
        if mask_a:
            target_tensor[drug_a_id] = 0
        if mask_b:
            target_tensor[drug_b_id] = 0
    else:
        old_a = target_tensor[drug_a_id, feat_idx].clone() if mask_a else None
        old_b = target_tensor[drug_b_id, feat_idx].clone() if mask_b else None
        if mask_a:
            target_tensor[drug_a_id, feat_idx] = 0
        if mask_b:
            target_tensor[drug_b_id, feat_idx] = 0
    try:
        perturbed_embedding = get_pair_fused_embedding(net, pair_tensor)
        rdfp_score = torch.norm(base_embedding - perturbed_embedding, p=2, dim=1).item()
    finally:
        if modality_name == "text_description" and feat_idx == -1:
            if mask_a:
                target_tensor[drug_a_id] = old_a
            if mask_b:
                target_tensor[drug_b_id] = old_b
        else:
            if mask_a:
                target_tensor[drug_a_id, feat_idx] = old_a
            if mask_b:
                target_tensor[drug_b_id, feat_idx] = old_b
    return rdfp_score

def plot_pair_topk_bar(top_df, fig_path, title):
    if top_df.empty:
        return
    plt.figure(figsize=(10, 5))
    draw_df = top_df.sort_values("rdfp", ascending=True)
    y_labels = draw_df["feature_name"].tolist()
    plt.barh(y_labels, draw_df["rdfp"], color="#2a9d8f")
    plt.xlabel("RDFP Distance (L2)")
    plt.title(title)
    for idx, v in enumerate(draw_df["rdfp"].tolist()):
        plt.text(v, idx, f" {v:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

def plot_label_combined_bar(label_top_df, fig_path, title):
    if label_top_df.empty:
        return
    draw_df = label_top_df.reset_index(drop=True).copy()
    draw_df["x_label"] = draw_df["feature_name"].astype(str)
    x = np.arange(len(draw_df))

    pair_ids = draw_df["sample_index"].astype(int).tolist()
    unique_pairs = []
    for pid in pair_ids:
        if pid not in unique_pairs:
            unique_pairs.append(pid)
    color_map = plt.cm.Set2(np.linspace(0, 1, max(len(unique_pairs), 2)))
    pair_to_color = {pid: color_map[i % len(color_map)] for i, pid in enumerate(unique_pairs)}
    bar_colors = [pair_to_color[pid] for pid in pair_ids]

    fig, ax = plt.subplots(figsize=(26, 8))
    bars = ax.bar(x, draw_df["rdfp"].values, color=bar_colors, edgecolor="#444444", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(draw_df["x_label"].tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RDFP Distance (L2)")
    ax.set_xlabel("Feature")
    ax.set_title(title)

    y_max = float(draw_df["rdfp"].max()) if len(draw_df) > 0 else 1.0
    y_upper = y_max * 1.35 if y_max > 0 else 1.0
    ax.set_ylim(0, y_upper)

    for rect, val in zip(bars, draw_df["rdfp"].values):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + y_upper * 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

    segment_start = 0
    current_pair = pair_ids[0]
    for i in range(1, len(pair_ids) + 1):
        is_break = (i == len(pair_ids)) or (pair_ids[i] != current_pair)
        if is_break:
            segment_end = i - 1
            segment_center = (segment_start + segment_end) / 2.0
            pair_df = draw_df.iloc[segment_start]
            pair_name = f"({pair_df['drugA_name']}-{pair_df['drugB_name']})"
            ax.text(
                segment_center,
                y_upper * 0.96,
                pair_name,
                ha="center",
                va="top",
                fontsize=9,
                color="#222222"
            )
            if i < len(pair_ids):
                ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.85)
                segment_start = i
                current_pair = pair_ids[i]

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

def run_fold_rdfp_analysis(
    net,
    x_datasets,
    x_test_indices,
    test_y,
    all_drugA_names,
    all_drugB_names,
    feature_name_maps,
    top_k,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    selected_positions = list(range(len(test_y)))
    if len(selected_positions) == 0:
        print("RDFP: test set is empty for current fold.")
        return

    net.eval()
    all_pair_top_records = []
    for pos in tqdm(selected_positions, desc="RDFP all test pairs"):
        sample_idx = int(x_test_indices[pos])
        label = int(test_y[pos])
        drug_a_id = int(x_datasets[sample_idx][0])
        drug_b_id = int(x_datasets[sample_idx][1])
        pair_tensor = torch.LongTensor([[drug_a_id, drug_b_id]]).to(device)
        base_embedding = get_pair_fused_embedding(net, pair_tensor)
        feature_specs = build_pair_feature_specs(net, drug_a_id, drug_b_id)

        pair_records = []
        for spec in feature_specs:
            modality_name = spec["modality"]
            feat_idx = spec["feature_index"]
            score = masked_feature_rdfp_score(
                net,
                pair_tensor,
                base_embedding,
                drug_a_id,
                drug_b_id,
                modality_name,
                feat_idx,
                spec["mask_a"],
                spec["mask_b"],
            )
            pair_records.append({
                "sample_index": sample_idx,
                "label": label,
                "drugA_name": str(all_drugA_names[sample_idx]),
                "drugB_name": str(all_drugB_names[sample_idx]),
                "drugA_id": drug_a_id,
                "drugB_id": drug_b_id,
                "modality": modality_name,
                "feature_index": feat_idx,
                "feature_source": spec["source"],
                "feature_name": resolve_feature_name(modality_name, feat_idx, feature_name_maps),
                "rdfp": score,
            })

        pair_df = pd.DataFrame(pair_records).sort_values("rdfp", ascending=False).reset_index(drop=True)
        pair_top_df = pair_df.head(top_k).copy()
        pair_top_df.insert(0, "rank", np.arange(1, len(pair_top_df) + 1))
        all_pair_top_records.extend(pair_top_df.to_dict(orient="records"))

    all_pair_top_df = pd.DataFrame(all_pair_top_records)
    if all_pair_top_df.empty:
        print("RDFP: no top-k records generated.")
        return

    all_pair_top_df = all_pair_top_df.sort_values(["sample_index", "rank"], ascending=[True, True]).reset_index(drop=True)
    all_pair_top_df.to_csv(
        os.path.join(output_dir, f"rdfp_all_test_pairs_top{top_k}.csv"),
        index=False
    )

    rank_count_df = (
        all_pair_top_df.groupby(["feature_name", "rank"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for r in range(1, top_k + 1):
        if r not in rank_count_df.columns:
            rank_count_df[r] = 0
    rank_count_df = rank_count_df[["feature_name"] + list(range(1, top_k + 1))]
    rename_map = {r: f"hit_top{r}" for r in range(1, top_k + 1)}
    rank_count_df = rank_count_df.rename(columns=rename_map)
    rank_count_df["hit_topk_total"] = rank_count_df[[f"hit_top{r}" for r in range(1, top_k + 1)]].sum(axis=1)
    rank_count_df = rank_count_df.sort_values(
        ["hit_topk_total"] + [f"hit_top{r}" for r in range(1, top_k + 1)],
        ascending=[False] + [False] * top_k
    ).reset_index(drop=True)
    rank_count_df.to_csv(
        os.path.join(output_dir, f"rdfp_feature_hit_counts_top{top_k}.csv"),
        index=False
    )

def main():
    # conn = sqlite3.connect(path_prefix + "/dataset/event.db")
    # df_drug = pd.read_sql('select * from drug;', conn)
    # extraction = pd.read_sql('select * from extraction;', conn)
    # mechanism = extraction['mechanism']
    # action = extraction['action']
    # drugA = extraction['drugA']
    # drugB = extraction['drugB']
    ddi_file_path = os.path.join(args.dataset_dir, args.ddi_file)
    drug_file_path = os.path.join(args.dataset_dir, args.drug_file)
    df_drug = pd.read_csv(drug_file_path)
    extraction = pd.read_csv(ddi_file_path)
    mechanism_action = extraction['Maps']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    new_label,event_num = prepare(mechanism_action)
    # new_label,event_num = prepare(mechanism, action)
    new_label = np.array(new_label)
    dict1 = {}
    for i in df_drug["drug_id"]:
        dict1[i] = len(dict1)

    drug_name = [dict1[i] for i in df_drug["drug_id"]]
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]
    n_drug = len(dict1)
    args.n_drug = n_drug
    dataset1_path = os.path.join(args.dataset_dir, args.dataset1_file)
    dataset2_path = os.path.join(args.dataset_dir, args.dataset2_file)
    dataset4_path = os.path.join(args.dataset_dir, args.dataset4_file)
    dataset1_kg, dataset1_tail_len, dataset1_relation_len, dataset1_tail_name_to_idx = read_dataset(dict1, dataset1_path)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len, dataset2_tail_name_to_idx = read_dataset(dict1, dataset2_path)
    # dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1,3)
    if os.path.exists(dataset4_path):
        dataset4_kg, dataset4_tail_len, dataset4_relation_len, dataset4_tail_name_to_idx = read_dataset(dict1, dataset4_path)
    else:
        print(f"Warning: {dataset4_path} not found. Use dataset1.txt as fallback for modality-4.")
        dataset4_kg, dataset4_tail_len, dataset4_relation_len, dataset4_tail_name_to_idx = read_dataset(dict1, dataset1_path)
    feature_name_maps = {
        "chemical_entities": invert_index_mapping(dataset1_tail_name_to_idx),
        "morgan_fingerprint": invert_index_mapping(dataset2_tail_name_to_idx),
        "maccs_fingerprint": invert_index_mapping(dataset4_tail_name_to_idx),
    }
    x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
    x_datasets = pd.DataFrame(data=x_datasets)
    x_datasets = x_datasets.to_numpy()
    dataset={}
    dataset["dataset1"],dataset["dataset2"],dataset["dataset4"] = dataset1_kg,dataset2_kg,dataset4_kg
    tail_len={}
    tail_len["dataset1"],tail_len["dataset2"],tail_len["dataset4"] = dataset1_tail_len,dataset2_tail_len,dataset4_tail_len
    relation_len={}
    relation_len["dataset1"],relation_len["dataset2"],relation_len["dataset4"] = dataset1_relation_len,dataset2_relation_len,dataset4_relation_len
    train_sum, test_sum = 0, 0

    feature_matrix_origin = torch.randn((n_drug+825, 256), dtype=float)
    # 药物与药物子结构的节点与特征
    feats = {}

    feats_1 = np.zeros((n_drug, dataset1_tail_len), dtype=float)
    feats_2 = np.zeros((n_drug, dataset2_tail_len), dtype=float)
    feats_4 = np.zeros((n_drug, dataset4_tail_len), dtype=float)

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
                    edge_index_1[1].append(n_drug + j[0])
                    real_edge_1.append(j[1])
                    num_classes_1.append(j[1])
                elif p == 1:
                    feats_2[i][j[0]] = 1
                    edge_index_2[0].append(i)
                    edge_index_2[1].append(n_drug + j[0])
                    real_edge_2.append(1)
                elif p == 2:
                    feats_4[i][j[0]] = j[1]
                    edge_index_4[0].append(i)
                    edge_index_4[1].append(n_drug + j[0])
                    real_edge_4.append(j[1])
                    num_classes_4.append(j[1])

    edge_index_1 = torch.tensor(edge_index_1)
    edge_index_2 = torch.tensor(edge_index_2)
    edge_index_4 = torch.tensor(edge_index_4)

    num_classes_1 = set(num_classes_1)
    num_classes_4 = set(num_classes_4)

    # from torch_geometric.data import HeteroData
    # data = HeteroData()

    # data['drug'].x = torch.randn(1710, 256)
    # data['chemical'].x = torch.randn(825, 256)

    # data['drug', 'interacts', 'chemical'].edge_index = torch.tensor(edge_index)
    # data['drug', 'interacts', 'chemical'].edge_attr = torch.tensor(edge_weight_tensor).view(-1, 1)

    # print(data)
    # feats_1 = torch.tensor(feats_1)

    # drug_idx, chemical_idx = torch.nonzero(feats_1, as_tuple=True)
    # chemical_idx += 1710
    # edge_index = torch.stack([drug_idx, chemical_idx], dim=0)
    # edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # from torch_geometric.datasets import Planetoid
    # import torch_geometric.transforms as T
    # from torch_geometric.utils import train_test_split_edges
    # dataset = Planetoid("/Users/charly/Downloads/MKG-FENN-main/Code and Datasets/data/data/", "CiteSeer", transform=T.NormalizeFeatures())
    # print(dataset.data)

    # data = dataset[0]
    # data.train_mask = data.val_mask = data.test_mask = data.y = None
    # data = train_test_split_edges(data)


    # feature_matrix3 = np.zeros((1710, 1710), dtype=float)
    feature_matrix1 = np.zeros((n_drug, dataset1_tail_len), dtype=float)
    feature_matrix2 = np.zeros((n_drug, dataset2_tail_len), dtype=float)
    feature_matrix3 = np.zeros((n_drug, dataset4_tail_len), dtype=float)
    feature_matrix4 = np.zeros((n_drug, n_drug), dtype=float)

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

    drug_sim1 = Jaccard(feature_matrix1)
    # feature_matrix2 = np.mat(feature_matrix2)
    drug_sim2 = Jaccard(feature_matrix2)
    # feature_matrix4 = np.mat(feature_matrix4)
    drug_sim4 = Jaccard(feature_matrix4)

    drug_sim3 = find_dif(feature_matrix3)


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
    for i in range(5):
        for k in range(4):
            for j in test_drug[i]:
                if k == 0:
                    target_list = drug_sim1[j].tolist()
                    max_v = 0
                    current_p = []
                    for p,v in enumerate(target_list[0]):
                        if v > max_v and p not in test_drug[i] and p != j:
                            max_v = v
                            current_p = []
                            current_p.append(p)
                        elif v == max_v and p not in test_drug[i] and p != j:
                            current_p.append(p)
                    test_adj[i][k][j].append(current_p)
                if k == 1:
                    target_list = drug_sim2[j].tolist()
                    max_v = 0
                    current_p = []
                    for p,v in enumerate(target_list[0]):
                        if v > max_v and p not in test_drug[i] and p != j:
                            max_v = v
                            current_p = []
                            current_p.append(p)
                        elif v == max_v and p not in test_drug[i] and p != j:
                            current_p.append(p)
                    test_adj[i][k][j].append(current_p)
                if k == 2:
                    target_list = drug_sim3[j].tolist()
                    max_v = 0
                    current_p = []
                    for p,v in enumerate(target_list):
                        if v > max_v and p not in test_drug[i] and p != j:
                            max_v = v
                            current_p = []
                            current_p.append(p)
                        elif v == max_v and p not in test_drug[i] and p != j:
                            current_p.append(p)
                    test_adj[i][k][j].append(current_p)
                if k == 3:
                    target_list = drug_sim4[j].tolist()
                    max_v = 0
                    current_p = []
                    for p, v in enumerate(target_list[0]):
                        if v > max_v and p not in test_drug[i] and p != j:
                            max_v = v
                            current_p = []
                            current_p.append(p)
                        elif v == max_v and p not in test_drug[i] and p != j:
                            current_p.append(p)
                    test_adj[i][k][j].append(current_p)
    y_true = np.array([])
    y_score = np.zeros((0, 86), dtype=float)
    y_pred = np.array([])
    tokenizer = ''
    
    # tokenizer = BertTokenizer.from_pretrained(path_prefix + "/model/chainese_bert")
    # model = BertModel.from_pretrained(path_prefix + "/model/chainese_bert")
    # each_drug_desc = get_desc_list2(dict1)
    # drug_emb = torch.tensor([])
    # for i in tqdm(each_drug_desc):
    #     bert_i, attention_mask = get_bert_encoding_with_sliding_window(i, tokenizer, model, 512, 512)
    #     drug_emb = torch.cat([drug_emb, get_text_embedding(bert_i, pool_strategy='mean')], 0)

    # drug_emb = torch.tensor(drug_emb)

    # drug_emb = torch.load(path_prefix + '/code/biobert_large_drug_emb.pt')

    # drug_emb2 = torch.load(path_prefix + '/code/biobert_large_drug_emb1.pt')

    # drug_emb = torch.load(path_prefix + '/drugdata/drug_emb.pt')


    # inputs = tokenizer(each_drug_desc, return_tensors='pt', padding='max_length', truncation=True, max_length=202, return_attention_mask=True)

    # 将输入移动到GPU（如果可用）
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    # model.eval() # 切换到评估模式

    # 获取模型的输出
    # with torch.no_grad():
        # outputs = model(**inputs)

    # 获取最后一层的隐藏状态
    # last_hidden_states = outputs.last_hidden_state

    # 输出句子的embedding（CLS token 的表示）
    # drug_emb = last_hidden_states[:, 0, :].squeeze()

    # drug_emb = encode(each_drug_desc)
    real_edge_1 = torch.tensor(real_edge_1)
    real_edge_2 = torch.tensor(real_edge_2)
    real_edge_4 = torch.tensor(real_edge_4)
    drug_emb = torch.load(args.drug_emb_path)
    rdfp_done = False
    # torch.save(drug_emb, '/home/jy/桌面/MKG/result4/100_baike_duiqi_tip_newmodel/drug_emb.pt')
    for main_l in [1]:
        for l in [0.06]:
            for l2 in [0.0000000000000001]:
                for cross_ver in range(2, 5):
                    # net = torch.load("/home/jy/桌面/MKG/MSREN-1710-rdfp/outputs/model_90.pth", weights_only=False)
                    # net = nn.Sequential(GNN1(dataset, tail_len, relation_len, args, dict1, drug_name, test_adj[cross_ver]),
                    #                     GNN2(dataset, tail_len, relation_len, args, dict1, drug_name, test_adj[cross_ver]),
                    #                     # GNN4(dataset, tail_len, relation_len, args, dict1, drug_name, test_adj[cross_ver]),
                    #                     # BertMLP(pathprefix + "/model/chainese_bert"),
                    #                     FusionLayer(args, test_adj[cross_ver]))

                    # kg1
                    net = nn.Sequential(Model1(args, dataset1_tail_len, feats_1, edge_index_1, 512, args.embedding_num, args.embedding_num, args.embedding_num, len(num_classes_1), real_edge_1),
                                        Model2(args, dataset2_tail_len, feats_2, edge_index_2, 512, args.embedding_num, args.embedding_num, args.embedding_num, 2, real_edge_2),
                                        Model4(args, dataset4_tail_len, feats_4, edge_index_4, 512, args.embedding_num, args.embedding_num, args.embedding_num, len(num_classes_4), real_edge_4),
                                        BertMLP(drug_emb),
                                        FusionLayer1(args))

                    # # bert_model = BertMLM(r"../model/base-uncased")
                    
                    # # net = nn.Sequential(bert_model, FusionLayer(args))
                    X_train = []
                    X_test = []
                    y_train = []
                    y_test = []
                    for i in range(len(drugA)):
                        if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                            X_train.append(i)
                            y_train.append(i)

                        # if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] not in np.array(train_drug[cross_ver])):
                        #     X_test.append(i)
                        #     y_test.append(i)
                        
                        # if (drugA_id[i] not in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                        #     X_test.append(i)
                        #     y_test.append(i)

                        if (drugA_id[i] not in np.array(train_drug[cross_ver])) and (drugB_id[i] not in np.array(train_drug[cross_ver])):
                            X_test.append(i)
                            y_test.append(i)


                    train_x = x_datasets[X_train]
                    train_y = new_label[y_train]
                    test_x = x_datasets[X_test]
                    test_y = new_label[y_test]

                    # test_loss, test_acc, train_loss, train_acc, test_list, test_output = train_gpu(train_x, train_y, test_x, test_y,
                    #                                                                            net,test_adj[cross_ver], dict1, tokenizer, drug_emb, sub_list=[])

                    test_loss, test_acc, train_loss, train_acc, test_list, test_output = train_gpu(train_x, train_y, test_x, test_y,
                                                                                            net,test_adj[cross_ver], dict1, tokenizer, drug_emb, real_edge_1, real_edge_2, real_edge_4, feats_1, feats_2, feats_4, main_l, l, l2)
                    # torch.save(net, "/home/jy/桌面/MKG/MSREN-1710-rdfp/outputs/model_90.pth")                                                                                            
                    # train_sum += train_acc
                    # test_sum += test_acc
                    # test_output = test_output.to('cpu')
                    # pred_type = torch.argmax(test_output, dim=1).numpy()
                    # y_pred = np.hstack((y_pred, pred_type))
                    # y_score = np.row_stack((y_score, test_output))
                    # y_true = np.hstack((y_true, test_y))
                    # print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
                    #         cross_ver, test_loss, test_acc, train_loss, train_acc))
                    if (
                        args.run_rdfp == 1
                        and (not rdfp_done)
                        and cross_ver == args.rdfp_fold
                    ):
                        run_fold_rdfp_analysis(
                            net=net,
                            x_datasets=x_datasets,
                            x_test_indices=X_test,
                            test_y=test_y,
                            all_drugA_names=drugA,
                            all_drugB_names=drugB,
                            feature_name_maps=feature_name_maps,
                            top_k=args.rdfp_top_k,
                            output_dir=args.rdfp_output_dir,
                        )
                        rdfp_done = True

                result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
                os.makedirs(args.result_dir, exist_ok=True)
                save_result(args.result_dir + "/loss_mlp_" + str(main_l) + '_' + str(l) + '_' + str(l2) + '_no_reloss_', "all", result_all)
                save_result(args.result_dir + "/loss_mlp_" + str(main_l) + '_' + str(l) + '_no_reloss_', "each", result_eve)
                print('%d-fold validation: avg train acc  %f, avg test acc %f' % (5, train_sum / 5, test_sum / 5))
    return

if __name__ == '__main__':
    main()
