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
import hashlib

# from long_code import *
import torch
from transformers import AutoTokenizer, BloomModel, AutoModel

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
import json
import re



path_prefix = r'/home/jy/桌面/MSE-ENN/'

# ========== 目标特征定义 ==========
DRUG_ID = 'D0349'
DRUG_NAME = 'clarithromycin'

# 目标特征（需要根据实际情况修改）
TARGET_ENZYMES = ["Cytochrome P450 3A4", "Cytochrome P450 2C9", "Cytochrome P450 3A", "Cytochrome P450 2D6", "Cytochrome P450 1A2", "Cytochrome P450 2C19", "Cytochrome P450 3A5"]
TARGET_TRANSPORTERS = ['OATP1B3', 'ABCB1']
TARGET_TARGETS = ['Vaccine/toxoid']
# 目标子结构（morgan id，来自 mecddi dataset2 的 tail）
TARGET_SUBSTRUCTURES = ["139", "140", "317", "560", "881"]
TARGET_MACCS = ["MACCS74", "MACCS85", "MACCS86", "MACCS93", "MACCS122", "MACCS148", "MACCS158"]

drug_pairs_file = path_prefix + 'Dataset2/drug_pairs_batch/' + DRUG_NAME + '_drug_pairs.csv'

EXPLAIN_MC_SAMPLES_MODALITY = 8
EXPLAIN_MC_SAMPLES_FEATURE = 4
EXPLAIN_MASK_VALUE = -1.0
EXPLAIN_BERT_MASK_VALUE = 0.0
BERT_TEXT_SOURCE_FILE = path_prefix + 'Dataset2/drug_matched.tsv'


# baike_duiqi = pd.read_excel(path_prefix + '/drugdata/baike_duiqi_original.xlsx', sheet_name="baike_duiqi_info_zh")


# 'cuda' if torch.cuda.is_available() else 
device = torch.device('cuda:1')

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches",type=int,choices=[100,500,1000,2000],default=1)
parser.add_argument("--batch_size",type=int,choices=[2048,1024,512,256,128],default=1024)
parser.add_argument("--weigh_decay",type=float,choices=[1e-1,1e-2,1e-3,1e-4,1e-8],default=1e-8)
parser.add_argument("--lr",type=float,choices=[1e-3,1e-4,1e-5,6*1e-3],default=5*1e-4) #5*1e-4 1e-4
# 对于不同的数据可以尝试不同采样数量
parser.add_argument("--neighbor_sample_size",choices=[4,6,10,16],type=int,default=6)
parser.add_argument("--event_num",type=int,default=47)

parser.add_argument("--n_drug",type=int,default=1621)
parser.add_argument("--seed",type=int,default=1)
parser.add_argument("--dropout",type=float,default=0.5)
parser.add_argument("--embedding_num",type=int,choices=[128,64,256,32],default=384)
parser.add_argument("--sample",type=int,choices=[1,2,3,4,5,6],default=1)
parser.add_argument("--main_l",type=float,default=1) #5*1e-4 1e-4
parser.add_argument("--second_l",type=float,default=0.08) #5*1e-4 1e-4
parser.add_argument("--explain_stage1_mask_values", type=str, default="-1")
parser.add_argument("--bert_text_source_file", type=str, default=BERT_TEXT_SOURCE_FILE)
parser.add_argument("--bert_model_path", type=str, default="/data/xhc/udever")
parser.add_argument("--bert_semantic_tags", type=str,
                    default="敏感菌,菌,呼吸道,皮肤,软组织,感染,非结核分枝杆菌,幽门螺杆菌")
parser.add_argument("--bert_max_sentences_per_side", type=int, default=6)
parser.add_argument("--bert_sentence_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
parser.add_argument("--bert_sentence_batch_size", type=int, default=8)
parser.add_argument("--top2_mode", type=str, default="union", choices=["union", "strict"])
parser.add_argument("--explain_feature_mask_values_model1", type=str, default="-1")
parser.add_argument("--explain_feature_mask_values_model2", type=str, default="-1")
parser.add_argument("--explain_feature_mask_values_model4", type=str, default="-1")

args = parser.parse_args()


def parse_float_list(text):
    vals = []
    for x in str(text).split(','):
        x = x.strip()
        if x == '':
            continue
        vals.append(float(x))
    return vals if len(vals) > 0 else [0.0]


def parse_str_list(text):
    vals = []
    for x in str(text).split(','):
        x = x.strip()
        if x:
            vals.append(x)
    return vals


EXPLAIN_STAGE1_MASK_VALUES = parse_float_list(args.explain_stage1_mask_values)
BERT_SEMANTIC_TAGS = [x.lower() for x in parse_str_list(args.bert_semantic_tags)]
EXPLAIN_FEATURE_MASK_VALUES_BY_MODALITY = {
    "Model1-Model1": parse_float_list(args.explain_feature_mask_values_model1),
    "Model2-Model1": parse_float_list(args.explain_feature_mask_values_model2),
    "Model4-Model1": parse_float_list(args.explain_feature_mask_values_model4),
}

def make_stage1_mask_rng(base_seed, fold_id, mask_value):
    key = f"{int(base_seed)}|{int(fold_id)}|{float(mask_value):.8f}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    seed = int(h[:16], 16)
    return random.Random(seed)

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


def read_dataset(drug_name_id,num):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list=[]
    filename = path_prefix + "/Dataset2/dataset"+str(num)+".txt"
    with open(filename, encoding="utf8") as reader:
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
            if num==3:
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    return kg,len(tails),len(relations)

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
        y_true_i = y_one_hot.take([i], axis=1).ravel()
        pred_i = pred_one_hot.take([i], axis=1).ravel()
        result_eve[i, 0] = accuracy_score(y_true_i, pred_i)
        result_eve[i, 1] = roc_aupr_score(y_true_i, pred_i, average=None)
        try:
            result_eve[i, 2] = roc_auc_score(y_true_i, pred_i, average=None)
        except ValueError:
            result_eve[i, 2] = 0.0
        result_eve[i, 3] = f1_score(y_true_i, pred_i, average='binary')
        result_eve[i, 4] = precision_score(y_true_i, pred_i, average='binary')
        result_eve[i, 5] = recall_score(y_true_i, pred_i, average='binary')
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

# tokenizer = AutoTokenizer.from_pretrained('/home/jy/桌面/MSE-ENN/model/udever')
# model = BloomModel.from_pretrained('/home/jy/桌面/MSE-ENN/model/udever')

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


def build_tail_map(num):
    tails = {}
    filename = path_prefix + "/Dataset2/dataset" + str(num) + ".txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string = line.rstrip().split('//', 2)
            if len(string) < 2:
                continue
            tail = string[1]
            if tail not in tails:
                tails[tail] = len(tails)
    return tails


def invert_map(d):
    return {v: k for k, v in d.items()}


def load_pairs_for_explain(dict1):
    if not os.path.exists(drug_pairs_file):
        return []
    df = pd.read_csv(drug_pairs_file)
    cols = list(df.columns)
    if len(cols) < 2:
        return []

    victim_col = None
    anchor_col = None
    for c in cols:
        if c.lower() == "victim_d_id":
            victim_col = c
            break
    for c in cols:
        if c.endswith("_D_ID") and c != victim_col:
            anchor_col = c
            break
    if anchor_col is None:
        anchor_col = cols[0]
    if victim_col is None:
        victim_col = cols[1]

    pairs = []
    for _, row in df.iterrows():
        a_id = str(row[anchor_col]).strip()
        b_id = str(row[victim_col]).strip()
        if a_id in dict1 and b_id in dict1:
            pairs.append((dict1[a_id], dict1[b_id], a_id, b_id))
    return pairs


def get_target_present_feats(feat_row, target_names, tail_map):
    out = []
    for name in target_names:
        idx = tail_map.get(name)
        if idx is not None and idx < len(feat_row) and feat_row[idx] != 0:
            out.append((name, idx))
    return out


def forward_pair_target_prob(net, pair_tensor, target_class):
    with torch.inference_mode():
        output = net(pair_tensor)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        probs = F.softmax(logits, dim=1)
        return float(probs[0, target_class].item())


def apply_ops_and_score(net, pair_tensor, target_class, ops, mask_val=EXPLAIN_MASK_VALUE, bert_mask=EXPLAIN_BERT_MASK_VALUE):
    drug_a = int(pair_tensor[0, 0].item())
    drug_b = int(pair_tensor[0, 1].item())

    backups = {}
    model_map = {
        'model1': net[0].autoencoder,
        'model2': net[1].autoencoder,
        'model4': net[2].autoencoder,
    }

    def save_row_col_once(autoencoder, key_prefix, drug_idx):
        row_key = (key_prefix, "drug", drug_idx)
        col_key = (key_prefix, "chem", drug_idx)
        if row_key not in backups:
            backups[row_key] = autoencoder.drug_features[drug_idx].clone()
        if col_key not in backups:
            backups[col_key] = autoencoder.chemical_features[:, drug_idx].clone()

    for op in ops:
        op_type = op['type']
        which_drug = drug_a if op['drug'] == 'a' else drug_b
        if op_type == 'feature':
            auto = model_map[op['model']]
            save_row_col_once(auto, op['model'], which_drug)
            if op.get('feat_idx') is None:
                auto.drug_features[which_drug].fill_(mask_val)
                auto.chemical_features[:, which_drug].fill_(mask_val)
            else:
                fi = int(op['feat_idx'])
                auto.drug_features[which_drug, fi] = mask_val
                auto.chemical_features[fi, which_drug] = mask_val
        elif op_type == 'bert':
            bert_key = ("bert", which_drug)
            if bert_key not in backups:
                backups[bert_key] = net[3].drug_emb[which_drug].clone()
            net[3].drug_emb[which_drug].fill_(bert_mask)

    score = forward_pair_target_prob(net, pair_tensor, target_class)

    for key, backup_tensor in backups.items():
        if key[0] in ('model1', 'model2', 'model4'):
            auto = model_map[key[0]]
            if key[1] == "drug":
                auto.drug_features[key[2]] = backup_tensor
            else:
                auto.chemical_features[:, key[2]] = backup_tensor
        else:
            net[3].drug_emb[key[1]] = backup_tensor
    return score


def build_modality_ops(modality_key):
    if modality_key == "Model1-Model1":
        return [
            {'type': 'feature', 'model': 'model1', 'drug': 'a', 'feat_idx': None},
            {'type': 'feature', 'model': 'model1', 'drug': 'b', 'feat_idx': None},
        ]
    if modality_key == "Model2-Model1":
        return [
            {'type': 'feature', 'model': 'model2', 'drug': 'a', 'feat_idx': None},
            {'type': 'feature', 'model': 'model1', 'drug': 'b', 'feat_idx': None},
        ]
    if modality_key == "Model4-Model1":
        return [
            {'type': 'feature', 'model': 'model4', 'drug': 'a', 'feat_idx': None},
            {'type': 'feature', 'model': 'model1', 'drug': 'b', 'feat_idx': None},
        ]
    if modality_key == "BertMLP-BertMLP":
        return [
            {'type': 'bert', 'drug': 'a'},
            {'type': 'bert', 'drug': 'b'},
        ]
    return []


def build_feature_ops(feature_item):
    if feature_item['modality'] == "Model1-Model1":
        return [
            {'type': 'feature', 'model': 'model1', 'drug': 'a', 'feat_idx': feature_item['a_idx']},
            {'type': 'feature', 'model': 'model1', 'drug': 'b', 'feat_idx': feature_item['b_idx']},
        ]
    if feature_item['modality'] == "Model2-Model1":
        return [
            {'type': 'feature', 'model': 'model2', 'drug': 'a', 'feat_idx': feature_item['a_idx']},
            {'type': 'feature', 'model': 'model1', 'drug': 'b', 'feat_idx': feature_item['b_idx']},
        ]
    if feature_item['modality'] == "Model4-Model1":
        return [
            {'type': 'feature', 'model': 'model4', 'drug': 'a', 'feat_idx': feature_item['a_idx']},
            {'type': 'feature', 'model': 'model1', 'drug': 'b', 'feat_idx': feature_item['b_idx']},
        ]
    return []


def apply_ops_and_score_batch_union(net, pair_tensor, target_classes, ops_by_pair, mask_val=EXPLAIN_MASK_VALUE, bert_mask=EXPLAIN_BERT_MASK_VALUE):
    backups = {}
    model_map = {
        'model1': net[0].autoencoder,
        'model2': net[1].autoencoder,
        'model4': net[2].autoencoder,
    }

    def save_row_col_once(autoencoder, key_prefix, drug_idx):
        row_key = (key_prefix, "drug", int(drug_idx))
        col_key = (key_prefix, "chem", int(drug_idx))
        if row_key not in backups:
            backups[row_key] = autoencoder.drug_features[int(drug_idx)].clone()
        if col_key not in backups:
            backups[col_key] = autoencoder.chemical_features[:, int(drug_idx)].clone()

    pair_np = pair_tensor.detach().cpu().numpy()
    for pair_i, ops in enumerate(ops_by_pair):
        drug_a = int(pair_np[pair_i, 0])
        drug_b = int(pair_np[pair_i, 1])
        for op in ops:
            op_type = op['type']
            which_drug = drug_a if op['drug'] == 'a' else drug_b
            if op_type == 'feature':
                auto = model_map[op['model']]
                save_row_col_once(auto, op['model'], which_drug)
                if op.get('feat_idx') is None:
                    auto.drug_features[which_drug].fill_(mask_val)
                    auto.chemical_features[:, which_drug].fill_(mask_val)
                else:
                    fi = int(op['feat_idx'])
                    auto.drug_features[which_drug, fi] = mask_val
                    auto.chemical_features[fi, which_drug] = mask_val
            elif op_type == 'bert':
                bert_key = ("bert", int(which_drug))
                if bert_key not in backups:
                    backups[bert_key] = net[3].drug_emb[int(which_drug)].clone()
                net[3].drug_emb[int(which_drug)].fill_(bert_mask)

    with torch.inference_mode():
        output = net(pair_tensor)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        probs = F.softmax(logits, dim=1)
        idx = torch.arange(probs.shape[0], device=probs.device)
        target_classes_t = torch.tensor(target_classes, dtype=torch.long, device=probs.device)
        scores = probs[idx, target_classes_t].detach().cpu().numpy()

    for key, backup_tensor in backups.items():
        if key[0] in ('model1', 'model2', 'model4'):
            auto = model_map[key[0]]
            if key[1] == "drug":
                auto.drug_features[key[2]] = backup_tensor
            else:
                auto.chemical_features[:, key[2]] = backup_tensor
        else:
            net[3].drug_emb[key[1]] = backup_tensor
    return scores


def build_core_feature_candidates_by_modality(drug_a, drug_b, core_modality, feats_1, feats_2, feats_4, dataset1_tails, dataset2_tails, dataset4_tails, dataset1_rel_map=None):
    if dataset1_rel_map is None:
        dataset1_rel_map = {}
    enz_a = dict(get_target_present_feats(feats_1[drug_a], TARGET_ENZYMES, dataset1_tails))
    enz_b = dict(get_target_present_feats(feats_1[drug_b], TARGET_ENZYMES, dataset1_tails))
    trp_a = dict(get_target_present_feats(feats_1[drug_a], TARGET_TRANSPORTERS, dataset1_tails))
    trp_b = dict(get_target_present_feats(feats_1[drug_b], TARGET_TRANSPORTERS, dataset1_tails))
    tar_a = dict(get_target_present_feats(feats_1[drug_a], TARGET_TARGETS, dataset1_tails))
    tar_b = dict(get_target_present_feats(feats_1[drug_b], TARGET_TARGETS, dataset1_tails))
    sub_a = dict(get_target_present_feats(feats_2[drug_a], TARGET_SUBSTRUCTURES, dataset2_tails))
    maccs_a = dict(get_target_present_feats(feats_4[drug_a], TARGET_MACCS, dataset4_tails))

    core_feature_candidates = []
    if core_modality == "Model1-Model1":
        for name in set(enz_a.keys()) & set(enz_b.keys()):
            role_pairs, rel_a, rel_b = get_role_pairs_for_feature(dataset1_rel_map, drug_a, drug_b, name, 'enzyme')
            # if len(role_pairs) == 0:
            #     continue
            core_feature_candidates.append({
                'modality': "Model1-Model1", 'rule': "TARGET_ENZYMES", 'a_name': name, 'b_name': name, 'a_idx': enz_a[name], 'b_idx': enz_b[name],
                'a_relations': rel_a, 'b_relations': rel_b, 'role_pairs': role_pairs
            })
        for name in set(trp_a.keys()) & set(trp_b.keys()):
            role_pairs, rel_a, rel_b = get_role_pairs_for_feature(dataset1_rel_map, drug_a, drug_b, name, 'transporter')
            # if len(role_pairs) == 0:
            #     continue
            core_feature_candidates.append({
                'modality': "Model1-Model1", 'rule': "TARGET_TRANSPORTERS", 'a_name': name, 'b_name': name, 'a_idx': trp_a[name], 'b_idx': trp_b[name],
                'a_relations': rel_a, 'b_relations': rel_b, 'role_pairs': role_pairs
            })
        for name in set(tar_a.keys()) & set(tar_b.keys()):
            role_pairs, rel_a, rel_b = get_role_pairs_for_feature(dataset1_rel_map, drug_a, drug_b, name, 'target')
            # if len(role_pairs) == 0:
            #     continue
            core_feature_candidates.append({
                'modality': "Model1-Model1", 'rule': "TARGET_TARGETS", 'a_name': name, 'b_name': name, 'a_idx': tar_a[name], 'b_idx': tar_b[name],
                'a_relations': rel_a, 'b_relations': rel_b, 'role_pairs': role_pairs
            })
    elif core_modality == "Model2-Model1":
        for s_name, s_idx in sub_a.items():
            for e_name, e_idx in enz_b.items():
                core_feature_candidates.append({
                    'modality': "Model2-Model1", 'rule': "TARGET_SUBSTRUCTURES_TO_TARGET_ENZYMES",
                    'a_name': s_name, 'b_name': e_name, 'a_idx': s_idx, 'b_idx': e_idx
                })
    elif core_modality == "Model4-Model1":
        for m_name, m_idx in maccs_a.items():
            for e_name, e_idx in enz_b.items():
                core_feature_candidates.append({
                    'modality': "Model4-Model1", 'rule': "TARGET_MACCS_TO_TARGET_ENZYMES",
                    'a_name': m_name, 'b_name': e_name, 'a_idx': m_idx, 'b_idx': e_idx
                })
    return core_feature_candidates


def load_pair_feature_role_map(dict1, drug_name):
    role_map = defaultdict(set)
    p = path_prefix + f'Dataset2/all_merged_pairs_batch/{drug_name}_all_merged_pairs.csv'
    if not os.path.exists(p):
        return role_map
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        return role_map
    need_cols = {'drugA_id', 'drugB_id', 'drugA_role', 'drugB_role'}
    if not need_cols.issubset(set(df.columns)):
        return role_map

    def norm_str(x):
        if pd.isna(x):
            return ''
        s = str(x).strip()
        if s.lower() == 'nan':
            return ''
        return s

    for _, row in df.iterrows():
        a_id = norm_str(row.get('drugA_id'))
        b_id = norm_str(row.get('drugB_id'))
        a_role = norm_str(row.get('drugA_role')).lower()
        b_role = norm_str(row.get('drugB_role')).lower()
        if a_id not in dict1 or b_id not in dict1:
            continue
        if a_role == '' or b_role == '':
            continue
        a_idx = int(dict1[a_id])
        b_idx = int(dict1[b_id])

        enz = norm_str(row.get('enzyme_name'))
        trp = norm_str(row.get('transporter_name'))
        tar = norm_str(row.get('factor_name'))
        if tar == '':
            tar = norm_str(row.get('target_name'))
        if enz != '':
            role_map[(a_idx, b_idx, normalize_feature_name(enz), 'enzyme')].add((a_role, b_role))
        if trp != '':
            role_map[(a_idx, b_idx, normalize_feature_name(trp), 'transporter')].add((a_role, b_role))
        if tar != '':
            role_map[(a_idx, b_idx, normalize_feature_name(tar), 'target')].add((a_role, b_role))
    return role_map


def compute_modality_shapley_for_pair(net, pair_tensor, target_class, rng):
    modalities = ["Model1-Model1", "Model2-Model1", "Model4-Model1", "BertMLP-BertMLP"]
    scores = {m: [] for m in modalities}
    for modality in modalities:
        others = [m for m in modalities if m != modality]
        for _ in range(EXPLAIN_MC_SAMPLES_MODALITY):
            k = rng.randint(0, len(others))
            subset = rng.sample(others, k) if k > 0 else []
            ops_t = []
            for md in subset:
                ops_t.extend(build_modality_ops(md))
            ops_ts = list(ops_t) + build_modality_ops(modality)
            f_t = apply_ops_and_score(net, pair_tensor, target_class, ops_t)
            f_ts = apply_ops_and_score(net, pair_tensor, target_class, ops_ts)
            scores[modality].append(f_t - f_ts)
    return {m: float(np.mean(v)) if len(v) > 0 else 0.0 for m, v in scores.items()}


def compute_feature_shapley_for_pair(net, pair_tensor, target_class, candidates, rng):
    if len(candidates) == 0:
        return []

    # 先做一次快速单特征对消融筛选，避免组合数过大导致解释时间过长
    base_score = apply_ops_and_score(net, pair_tensor, target_class, [])
    quick_scores = []
    for cand in candidates:
        masked_score = apply_ops_and_score(net, pair_tensor, target_class, build_feature_ops(cand))
        quick_scores.append((cand, base_score - masked_score))
    quick_scores.sort(key=lambda x: x[1], reverse=True)
    screened = [x[0] for x in quick_scores[:min(12, len(quick_scores))]]

    feature_scores = []
    for idx, cand in enumerate(screened):
        contribs = []
        other_idx = [i for i in range(len(screened)) if i != idx]
        for _ in range(EXPLAIN_MC_SAMPLES_FEATURE):
            k = rng.randint(0, len(other_idx))
            subset = rng.sample(other_idx, k) if k > 0 else []
            ops_t = []
            for oi in subset:
                ops_t.extend(build_feature_ops(screened[oi]))
            ops_ts = list(ops_t) + build_feature_ops(cand)
            f_t = apply_ops_and_score(net, pair_tensor, target_class, ops_t)
            f_ts = apply_ops_and_score(net, pair_tensor, target_class, ops_ts)
            contribs.append(f_t - f_ts)
        feature_scores.append({
            'modality': cand['modality'],
            'rule': cand['rule'],
            'feature_a': cand['a_name'],
            'feature_b': cand['b_name'],
            'a_relations': cand.get('a_relations', []),
            'b_relations': cand.get('b_relations', []),
            'role_pairs': cand.get('role_pairs', []),
            'shapley_value': float(np.mean(contribs)) if len(contribs) > 0 else 0.0
        })
    feature_scores.sort(key=lambda x: x['shapley_value'], reverse=True)
    return feature_scores


def check_hit_logic(modality_key, feature_item):
    if feature_item is None:
        return False
    fa = feature_item.get('feature_a')
    fb = feature_item.get('feature_b')
    a_rel = feature_item.get('a_relations', [])
    b_rel = feature_item.get('b_relations', [])
    if modality_key == "Model1-Model1":
        role_pairs = feature_item.get('role_pairs', [])
        if fa == fb and fa in TARGET_ENZYMES and len(role_pairs) > 0 and len(a_rel) > 0 and len(b_rel) > 0:
            return True
        if fa == fb and fa in TARGET_TRANSPORTERS and len(role_pairs) > 0 and len(a_rel) > 0 and len(b_rel) > 0:
            return True
        if fa == fb and fa in TARGET_TARGETS and len(role_pairs) > 0 and len(a_rel) > 0 and len(b_rel) > 0:
            return True
        return False
    if modality_key == "Model2-Model1":
        return (fa in TARGET_SUBSTRUCTURES) and (fb in TARGET_ENZYMES)
    if modality_key == "Model4-Model1":
        return (fa in TARGET_MACCS) and (fb in TARGET_ENZYMES)
    if modality_key == "BertMLP-BertMLP":
        return False
    return False


def check_model1_rule_hit(rule_key, feature_item):
    if feature_item is None:
        return False
    fa = feature_item.get('feature_a')
    fb = feature_item.get('feature_b')
    a_rel = feature_item.get('a_relations', [])
    b_rel = feature_item.get('b_relations', [])
    role_pairs = feature_item.get('role_pairs', [])
    if rule_key == "enzyme-enzyme":
        return (fa == fb) and (fa in TARGET_ENZYMES) and (len(role_pairs) > 0) and (len(a_rel) > 0) and (len(b_rel) > 0)
    if rule_key == "transporter-transporter":
        return (fa == fb) and (fa in TARGET_TRANSPORTERS) and (len(role_pairs) > 0) and (len(a_rel) > 0) and (len(b_rel) > 0)
    if rule_key == "target-target":
        return (fa == fb) and (fa in TARGET_TARGETS) and (len(role_pairs) > 0) and (len(a_rel) > 0) and (len(b_rel) > 0)
    return False


def split_sentences_for_bert(text):
    if text is None:
        return []
    s = str(text).strip()
    if s == '':
        return []
    parts = re.split(r'[。！？!?\.;；,，]', s)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def check_semantic_tags_for_sentence(sentence, tags):
    s = str(sentence).lower()
    found = []
    for t in tags:
        if t and t in s:
            found.append(t)
    return found


def safe_adjust_embedding_dim(arr_1d, target_dim):
    x = np.asarray(arr_1d, dtype=np.float32)
    if x.shape[0] == target_dim:
        return x
    if x.shape[0] > target_dim:
        return x[:target_dim]
    out = np.zeros((target_dim,), dtype=np.float32)
    out[:x.shape[0]] = x
    return out


def normalize_drug_name(name):
    if name is None:
        return ''
    return re.sub(r'\s+', ' ', str(name).strip().lower())


def normalize_feature_name(name):
    if name is None:
        return ''
    s = str(name).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('_', ' ').replace('-', ' ')
    s = re.sub(r'\s+', ' ', s).strip()

    # transporter aliases
    if s in {'p glycoprotein 1', 'p glycoprotein', 'pgp', 'p gp', 'mdr1', 'abc b1'}:
        return 'abcb1'

    # CYP aliases -> cytochrome p450 family canonical text
    m = re.match(r'^cyp(?:450)?\s*([0-9][a-z0-9]*)$', s)
    if m:
        return f'cytochrome p450 {m.group(1)}'

    return s


def get_role_pairs_for_feature(role_map, drug_a_idx, drug_b_idx, feature_name, feature_type):
    fn = normalize_feature_name(feature_name)
    pairs = role_map.get((int(drug_a_idx), int(drug_b_idx), fn, feature_type), set())
    if pairs:
        pairs_sorted = sorted(list(pairs))
        rel_a = sorted(list({x[0] for x in pairs_sorted}))
        rel_b = sorted(list({x[1] for x in pairs_sorted}))
        return pairs_sorted, rel_a, rel_b

    # 回退：若表内方向反了，交换role顺序后返回
    rev_pairs = role_map.get((int(drug_b_idx), int(drug_a_idx), fn, feature_type), set())
    if rev_pairs:
        swapped = sorted(list({(rb, ra) for (ra, rb) in rev_pairs}))
        rel_a = sorted(list({x[0] for x in swapped}))
        rel_b = sorted(list({x[1] for x in swapped}))
        return swapped, rel_a, rel_b

    return [], [], []


def load_drug_id_to_name_map():
    p = path_prefix + 'Dataset2/drug_smiles.csv'
    if not os.path.exists(p):
        return {}
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        return {}
    if 'drug_id' not in df.columns or 'drug_name' not in df.columns:
        return {}
    did = df['drug_id'].fillna('').astype(str).str.strip()
    dname = df['drug_name'].fillna('').astype(str).str.strip()
    valid = did.ne('') & dname.ne('') & dname.str.lower().ne('nan')
    # 与旧逻辑一致：重复drug_id时后者覆盖前者
    out = dict(zip(did[valid].tolist(), dname[valid].tolist()))
    return out


def load_drug_description_map(text_source_file):
    if not os.path.exists(text_source_file):
        return {}
    try:
        # 自动兼容 tsv/csv
        df = pd.read_csv(text_source_file, sep='\t', low_memory=False)
    except Exception:
        return {}
    if 'drug_name' not in df.columns or 'description' not in df.columns:
        return {}
    name = (
        df['drug_name']
        .fillna('')
        .astype(str)
        .str.lower()
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    desc = df['description'].fillna('').astype(str).str.strip()
    valid = name.ne('') & desc.ne('') & desc.str.lower().ne('nan')
    # 与旧逻辑一致：重复drug_name时后者覆盖前者
    out = dict(zip(name[valid].tolist(), desc[valid].tolist()))
    return out


def build_bert_text_sides_from_tsv(drug_a_id, drug_b_id, drug_id_to_name, desc_map):
    name_a = normalize_drug_name(drug_id_to_name.get(str(drug_a_id).strip(), ''))
    name_b = normalize_drug_name(drug_id_to_name.get(str(drug_b_id).strip(), ''))
    text_a = desc_map.get(name_a, '')
    text_b = desc_map.get(name_b, '')
    return text_a, text_b


def load_bert_sentence_encoder(model_path, device_for_model):
    if not os.path.exists(model_path):
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.to(device_for_model)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


def resolve_bert_sentence_device(local_device):
    mode = str(args.bert_sentence_device).lower().strip()
    if mode == 'cpu':
        return torch.device('cpu')
    if mode == 'cuda':
        return local_device if str(local_device).startswith('cuda') else torch.device('cpu')
    # auto: 优先跟随主模型设备
    return local_device if str(local_device).startswith('cuda') else torch.device('cpu')


def encode_sentences_with_model(sentences, tokenizer, model, device_for_model, target_dim, batch_size=16):
    if len(sentences) == 0:
        return np.zeros((0, target_dim), dtype=np.float32)
    if tokenizer is None or model is None:
        # fallback: 使用确定性哈希向量，避免全零导致shapley/prob_delta全为0
        outs = []
        for s in sentences:
            seed = abs(hash(str(s))) % (2 ** 32)
            rng_local = np.random.default_rng(seed)
            v = rng_local.standard_normal(target_dim).astype(np.float32)
            nrm = float(np.linalg.norm(v))
            if nrm > 0:
                v = v / nrm
            outs.append(v)
        return np.array(outs, dtype=np.float32)
    outs = []
    bs = max(1, int(batch_size))
    with torch.inference_mode():
        for i in range(0, len(sentences), bs):
            ss = sentences[i:i+bs]
            enc = tokenizer(
                ss,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors='pt'
            )
            enc = {k: v.to(device_for_model) for k, v in enc.items()}
            out = model(**enc).last_hidden_state[:, 0, :].detach().cpu().numpy()
            for row in out:
                outs.append(safe_adjust_embedding_dim(row, target_dim))
    return np.array(outs, dtype=np.float32)


def score_with_bert_sentence_pair(net, pair_tensor, target_class, emb_a, emb_b, bert_dim):
    drug_a = int(pair_tensor[0, 0].item())
    drug_b = int(pair_tensor[0, 1].item())
    bkp_a = net[3].drug_emb[drug_a].clone()
    bkp_b = net[3].drug_emb[drug_b].clone()
    try:
        net[3].drug_emb[drug_a] = torch.tensor(emb_a, dtype=net[3].drug_emb.dtype, device=net[3].drug_emb.device)
        net[3].drug_emb[drug_b] = torch.tensor(emb_b, dtype=net[3].drug_emb.dtype, device=net[3].drug_emb.device)
        return forward_pair_target_prob(net, pair_tensor, target_class)
    finally:
        net[3].drug_emb[drug_a] = bkp_a
        net[3].drug_emb[drug_b] = bkp_b


def calculate_sentence_pair_shapley_for_bert(net, pair_tensor, target_class, emb_a, emb_b, bert_dim):
    zero_vec = np.zeros((bert_dim,), dtype=np.float32)
    f_empty = score_with_bert_sentence_pair(net, pair_tensor, target_class, zero_vec, zero_vec, bert_dim)
    f_a = score_with_bert_sentence_pair(net, pair_tensor, target_class, emb_a, zero_vec, bert_dim)
    f_b = score_with_bert_sentence_pair(net, pair_tensor, target_class, zero_vec, emb_b, bert_dim)
    f_ab = score_with_bert_sentence_pair(net, pair_tensor, target_class, emb_a, emb_b, bert_dim)

    marginal_a = f_a - f_empty
    marginal_b = f_b - f_empty
    marginal_ab = f_ab - f_a - f_b + f_empty
    shapley_a = marginal_a + 0.5 * marginal_ab
    shapley_b = marginal_b + 0.5 * marginal_ab
    return float((shapley_a + shapley_b) / 2.0), float(f_ab - f_empty)


def run_explainability_after_training(net, dict1, feats_1, feats_2, feats_4, dataset1_tails, dataset2_tails, dataset4_tails, fold_id):
    save_dir = os.path.join(path_prefix, "result_Dataset2_task2", f"explain_{DRUG_ID}")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"fold{fold_id+1}_epoch{args.epoches}_pair_shapley.json")

    pairs = load_pairs_for_explain(dict1)
    if len(pairs) == 0:
        print(f"[Explain] 未从 {drug_pairs_file} 读取到可用药物对，跳过归因")
        return

    idx_to_id = invert_map(dict1)
    net.eval()
    local_device = next(net.parameters()).device
    rng = random.Random(args.seed + 1000 + fold_id)
    drug_id_to_name = load_drug_id_to_name_map()
    drug_desc_map = load_drug_description_map(args.bert_text_source_file)
    dataset1_rel_map = load_pair_feature_role_map(dict1, DRUG_NAME)
    bert_dim = int(net[3].drug_emb.shape[1])
    bert_sentence_emb_cache = {}
    results = []
    hit_stats = {
        'Model1-Model1': {'top1': 0, 'top2': 0},
        'Model2-Model1': {'top1': 0, 'top2': 0},
        'Model4-Model1': {'top1': 0, 'top2': 0},
        'BertMLP-BertMLP': {'top1': 0, 'top2': 0},
    }
    hit_stats_raw_top2 = {
        'Model1-Model1': 0,
        'Model2-Model1': 0,
        'Model4-Model1': 0,
        'BertMLP-BertMLP': 0,
    }
    cand_ge2_stats = {
        'Model1-Model1': 0,
        'Model2-Model1': 0,
        'Model4-Model1': 0,
        'BertMLP-BertMLP': 0,
    }
    hit_stats_model1_rules = {
        'enzyme-enzyme': {'top1': 0, 'top2': 0},
        'transporter-transporter': {'top1': 0, 'top2': 0},
        'target-target': {'top1': 0, 'top2': 0},
    }
    model1_top2_dedup_total = 0
    total_top1_hit = 0
    total_top2_hit = 0

    # 第一阶段：批量确定全部药物对的核心模态对（只在融合层做模态掩码，速度更快）
    stage1_pairs = []
    pair_arr = np.array([[int(a), int(b)] for a, b, _, _ in pairs], dtype=np.int64)
    pair_tensor_all = torch.tensor(pair_arr, dtype=torch.long, device=local_device)
    with torch.inference_mode():
        gae1_embedding, gae1_z, idx = net[0](pair_tensor_all)
        gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = net[1]((gae1_embedding, gae1_z, idx))
        gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = net[2]((gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx))
        llm_embedding, gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = net[3]((gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx))
        drugA = idx[:, 0]
        drugB = idx[:, 1]
        pair_embedding_all = torch.cat([
            gae1_embedding[drugA], gae2_embedding[drugA], gae4_embedding[drugA], llm_embedding[drugA],
            gae1_embedding[drugB], gae2_embedding[drugB], gae4_embedding[drugB], llm_embedding[drugB]
        ], dim=1).float()
        logits_all = net[4].fullConnectionLayer(pair_embedding_all)
        prob_all = F.softmax(logits_all, dim=1)
        target_classes_all = torch.argmax(prob_all, dim=1)
        target_probs_all = prob_all[torch.arange(prob_all.shape[0], device=local_device), target_classes_all]

    modalities = ["Model1-Model1", "Model2-Model1", "Model4-Model1", "BertMLP-BertMLP"]
    modality_seg_map = {
        "Model1-Model1": [0, 4],
        "Model2-Model1": [1, 4],
        "Model4-Model1": [2, 4],
        "BertMLP-BertMLP": [3, 7],
    }
    seg_dim = int(args.embedding_num)
    seg_ranges = [(i * seg_dim, (i + 1) * seg_dim) for i in range(8)]

    stage1_scan_results = []
    stage1_pairs_selected = None
    selected_mask_value = EXPLAIN_STAGE1_MASK_VALUES[0]

    for mask_value in EXPLAIN_STAGE1_MASK_VALUES:
        rng_mask = make_stage1_mask_rng(args.seed, fold_id, mask_value)
        stage1_cache = {}

        def stage1_scores_for_subset(subset_modalities):
            key = tuple(sorted(subset_modalities))
            if key in stage1_cache:
                return stage1_cache[key]
            masked_embed = pair_embedding_all.clone()
            for md in subset_modalities:
                for seg_idx in modality_seg_map[md]:
                    st, ed = seg_ranges[seg_idx]
                    masked_embed[:, st:ed] = float(mask_value)
            with torch.inference_mode():
                logits = net[4].fullConnectionLayer(masked_embed)
                probs = F.softmax(logits, dim=1)
                scores = probs[torch.arange(probs.shape[0], device=local_device), target_classes_all]
            stage1_cache[key] = scores.detach().cpu().numpy()
            return stage1_cache[key]

        modality_shapley_batch = {m: np.zeros(len(pairs), dtype=np.float32) for m in modalities}
        for modality in modalities:
            others = [m for m in modalities if m != modality]
            for _ in range(EXPLAIN_MC_SAMPLES_MODALITY):
                k = rng_mask.randint(0, len(others))
                subset = rng_mask.sample(others, k) if k > 0 else []
                scores_t = stage1_scores_for_subset(subset)
                scores_ts = stage1_scores_for_subset(subset + [modality])
                modality_shapley_batch[modality] += (scores_t - scores_ts)
            modality_shapley_batch[modality] /= float(EXPLAIN_MC_SAMPLES_MODALITY)

        stage1_pairs_cur = []
        stage1_modality_top1_hits = {
            'Model1-Model1': 0,
            'Model2-Model1': 0,
            'Model4-Model1': 0,
            'BertMLP-BertMLP': 0,
        }
        for pair_idx, (drug_a, drug_b, drug_a_id, drug_b_id) in enumerate(pairs):
            modality_shapley = {m: float(modality_shapley_batch[m][pair_idx]) for m in modalities}
            modality_rank = sorted(modality_shapley.items(), key=lambda x: x[1], reverse=True)
            top_modality = {
                'rank': 1,
                'modality': modality_rank[0][0] if len(modality_rank) > 0 else None,
                'shapley_value': float(modality_rank[0][1]) if len(modality_rank) > 0 else None
            }
            best_modality = top_modality['modality']
            if best_modality in stage1_modality_top1_hits:
                stage1_modality_top1_hits[best_modality] += 1

            stage1_pairs_cur.append({
                'pair_index': int(pair_idx),
                'drug_a_idx': int(drug_a),
                'drug_b_idx': int(drug_b),
                'drug_a_id': drug_a_id if drug_a_id else idx_to_id.get(drug_a, str(drug_a)),
                'drug_b_id': drug_b_id if drug_b_id else idx_to_id.get(drug_b, str(drug_b)),
                'target_class': int(target_classes_all[pair_idx].item()),
                'target_prob': float(target_probs_all[pair_idx].item()),
                'modality_shapley': modality_shapley,
                'top_modality': top_modality,
                'best_modality': best_modality,
                'stage1_mask_value': float(mask_value),
            })

        print(f"[Explain][Stage1] 掩码值 {mask_value} 的各模态对 top1 命中数:")
        for mod_name in ['Model1-Model1', 'Model2-Model1', 'Model4-Model1', 'BertMLP-BertMLP']:
            print(f"  {mod_name}: {stage1_modality_top1_hits.get(mod_name, 0)}")

        stage1_scan_results.append({
            'mask_value': float(mask_value),
            'modality_top1_hits': dict(stage1_modality_top1_hits)
        })
        if stage1_pairs_selected is None:
            stage1_pairs_selected = stage1_pairs_cur

    stage1_pairs = stage1_pairs_selected if stage1_pairs_selected is not None else []
    stage1_modality_top1_hits = stage1_scan_results[0]['modality_top1_hits'] if len(stage1_scan_results) > 0 else {
        'Model1-Model1': 0, 'Model2-Model1': 0, 'Model4-Model1': 0, 'BertMLP-BertMLP': 0
    }

    # 第二阶段：按核心模态分组批量计算特征对贡献（近似批量）
    results_by_pair_idx = {int(item['pair_index']): dict(item) for item in stage1_pairs}
    modality_order = ["Model1-Model1", "Model2-Model1", "Model4-Model1", "BertMLP-BertMLP"]
    for core_modality in modality_order:
        group_items = [x for x in stage1_pairs if x.get('best_modality') == core_modality]
        if len(group_items) == 0:
            continue

        if core_modality == "BertMLP-BertMLP":
            bert_device = resolve_bert_sentence_device(local_device)
            bert_tokenizer, bert_sentence_model = load_bert_sentence_encoder(args.bert_model_path, bert_device)
            if bert_tokenizer is None or bert_sentence_model is None:
                print(f"[Explain][BERT] 句子模型加载失败，降级为哈希向量编码: {args.bert_model_path}")
            for it in group_items:
                pair_tensor = torch.tensor([[int(it['drug_a_idx']), int(it['drug_b_idx'])]], dtype=torch.long, device=local_device)
                target_class = int(it['target_class'])
                drug_a_id = str(it.get('drug_a_id', ''))
                drug_b_id = str(it.get('drug_b_id', ''))
                text_a, text_b = build_bert_text_sides_from_tsv(
                    drug_a_id, drug_b_id, drug_id_to_name, drug_desc_map
                )
                sentences_a = split_sentences_for_bert(text_a)[:max(1, int(args.bert_max_sentences_per_side))]
                sentences_b = split_sentences_for_bert(text_b)[:max(1, int(args.bert_max_sentences_per_side))]

                sentence_pairs = []
                if len(sentences_a) > 0 and len(sentences_b) > 0:
                    all_sent = sentences_a + sentences_b
                    sent_to_emb = {}
                    miss = [s for s in all_sent if s not in bert_sentence_emb_cache]
                    if len(miss) > 0:
                        miss_emb = encode_sentences_with_model(
                            miss, bert_tokenizer, bert_sentence_model, bert_device, bert_dim,
                            batch_size=args.bert_sentence_batch_size
                        )
                        for s, e in zip(miss, miss_emb):
                            bert_sentence_emb_cache[s] = e
                    for s in all_sent:
                        sent_to_emb[s] = bert_sentence_emb_cache.get(s, np.zeros((bert_dim,), dtype=np.float32))
                    emb_a_mat = np.array([sent_to_emb[s] for s in sentences_a], dtype=np.float32)
                    emb_b_mat = np.array([sent_to_emb[s] for s in sentences_b], dtype=np.float32)
                    m = emb_a_mat.shape[0]
                    n = emb_b_mat.shape[0]

                    with torch.inference_mode():
                        zero_vec = np.zeros((1, bert_dim), dtype=np.float32)
                        raw_a = np.vstack([zero_vec, emb_a_mat])
                        raw_b = np.vstack([zero_vec, emb_b_mat])
                        raw_a_t = torch.tensor(raw_a, dtype=net[3].drug_emb.dtype, device=local_device)
                        raw_b_t = torch.tensor(raw_b, dtype=net[3].drug_emb.dtype, device=local_device)
                        proj_a = net[3].fullConnectionLayer(raw_a_t)  # [m+1, emb_dim]
                        proj_b = net[3].fullConnectionLayer(raw_b_t)  # [n+1, emb_dim]

                        da = int(it['drug_a_idx'])
                        db = int(it['drug_b_idx'])
                        g1a = gae1_embedding[da].unsqueeze(0)
                        g2a = gae2_embedding[da].unsqueeze(0)
                        g4a = gae4_embedding[da].unsqueeze(0)
                        g1b = gae1_embedding[db].unsqueeze(0)
                        g2b = gae2_embedding[db].unsqueeze(0)
                        g4b = gae4_embedding[db].unsqueeze(0)

                        def fusion_target_scores(llm_a_batch, llm_b_batch):
                            bsz = llm_a_batch.shape[0]
                            x = torch.cat([
                                g1a.repeat(bsz, 1), g2a.repeat(bsz, 1), g4a.repeat(bsz, 1), llm_a_batch,
                                g1b.repeat(bsz, 1), g2b.repeat(bsz, 1), g4b.repeat(bsz, 1), llm_b_batch
                            ], dim=1).float()
                            logits = net[4].fullConnectionLayer(x)
                            probs = F.softmax(logits, dim=1)
                            return probs[:, target_class]

                    candidates = []
                    for i, sa in enumerate(sentences_a):
                        for j, sb in enumerate(sentences_b):
                            tags_a = check_semantic_tags_for_sentence(sa, BERT_SEMANTIC_TAGS)
                            tags_b = check_semantic_tags_for_sentence(sb, BERT_SEMANTIC_TAGS)
                            candidates.append({
                                'idx_a': int(i),
                                'idx_b': int(j),
                                'feature_a': sa,
                                'feature_b': sb,
                                'tags_a': tags_a,
                                'tags_b': tags_b,
                                'both_have_tags': bool(len(tags_a) > 0 and len(tags_b) > 0),
                            })

                    subset_score_cache = {}
                    subset_embed_cache = {}

                    def get_subset_embed(key):
                        if key in subset_embed_cache:
                            return subset_embed_cache[key]
                        if len(key) == 0:
                            llm_a = proj_a[0:1]
                            llm_b = proj_b[0:1]
                        else:
                            a_set = sorted(set([candidates[k]['idx_a'] for k in key]))
                            b_set = sorted(set([candidates[k]['idx_b'] for k in key]))
                            a_idx_t = torch.tensor([x + 1 for x in a_set], device=local_device, dtype=torch.long)
                            b_idx_t = torch.tensor([x + 1 for x in b_set], device=local_device, dtype=torch.long)
                            llm_a = torch.mean(proj_a[a_idx_t], dim=0, keepdim=True)
                            llm_b = torch.mean(proj_b[b_idx_t], dim=0, keepdim=True)
                        subset_embed_cache[key] = (llm_a, llm_b)
                        return llm_a, llm_b

                    def eval_subset_keys_batch(keys, batch_size=256):
                        miss = [k for k in keys if k not in subset_score_cache]
                        if len(miss) == 0:
                            return
                        for st in range(0, len(miss), batch_size):
                            part = miss[st:st + batch_size]
                            llm_a_batch = []
                            llm_b_batch = []
                            for key in part:
                                ea, eb = get_subset_embed(key)
                                llm_a_batch.append(ea.squeeze(0))
                                llm_b_batch.append(eb.squeeze(0))
                            llm_a_batch = torch.stack(llm_a_batch, dim=0)
                            llm_b_batch = torch.stack(llm_b_batch, dim=0)
                            vals = fusion_target_scores(llm_a_batch, llm_b_batch).detach().cpu().numpy()
                            for i_key, key in enumerate(part):
                                subset_score_cache[key] = float(vals[i_key])

                    empty_key = tuple()
                    eval_subset_keys_batch([empty_key], batch_size=32)
                    empty_score = subset_score_cache[empty_key]

                    for ci, cand in enumerate(candidates):
                        others = [x for x in range(len(candidates)) if x != ci]
                        sampled_keys_t = []
                        sampled_keys_ts = []
                        keys_need = {empty_key}
                        for _ in range(EXPLAIN_MC_SAMPLES_FEATURE):
                            k = rng.randint(0, len(others))
                            subset = rng.sample(others, k) if k > 0 else []
                            key_t = tuple(sorted(subset))
                            key_ts = tuple(sorted(subset + [ci]))
                            sampled_keys_t.append(key_t)
                            sampled_keys_ts.append(key_ts)
                            keys_need.add(key_t)
                            keys_need.add(key_ts)

                        eval_subset_keys_batch(list(keys_need))

                        contribs = []
                        for key_t, key_ts in zip(sampled_keys_t, sampled_keys_ts):
                            contribs.append(subset_score_cache[key_ts] - subset_score_cache[key_t])

                        key_single = (ci,)
                        if key_single not in subset_score_cache:
                            eval_subset_keys_batch([key_single], batch_size=32)
                        prob_delta = subset_score_cache[key_single] - empty_score
                        sentence_pairs.append({
                            'idx_a': cand['idx_a'],
                            'idx_b': cand['idx_b'],
                            'feature_a': cand['feature_a'],
                            'feature_b': cand['feature_b'],
                            'shapley_value': float(np.mean(contribs)) if len(contribs) > 0 else 0.0,
                            'prob_delta': float(prob_delta),
                            'tags_a': cand['tags_a'],
                            'tags_b': cand['tags_b'],
                            'both_have_tags': cand['both_have_tags'],
                        })

                    sentence_pairs.sort(key=lambda x: x['shapley_value'], reverse=True)

                top1_feat = sentence_pairs[0] if len(sentence_pairs) > 0 else None
                top2_feat = sentence_pairs[1] if len(sentence_pairs) > 1 else None
                top1_hit = bool(top1_feat['both_have_tags']) if top1_feat is not None else False
                top2_raw_hit = bool(top2_feat['both_have_tags']) if top2_feat is not None else False
                top2_hit = bool(top2_raw_hit) if args.top2_mode == "strict" else bool(top1_hit or top2_raw_hit)
                if len(sentence_pairs) >= 2:
                    cand_ge2_stats[core_modality] += 1
                if top2_raw_hit:
                    hit_stats_raw_top2[core_modality] += 1
                if top1_hit:
                    hit_stats[core_modality]['top1'] += 1
                    total_top1_hit += 1
                if top2_hit:
                    hit_stats[core_modality]['top2'] += 1
                    total_top2_hit += 1

                feature_details = {
                    'core_modality': core_modality,
                    'candidate_count': len(sentence_pairs),
                    'top1': top1_feat,
                    'top2': top2_feat,
                    'top1_hit': bool(top1_hit),
                    'top2_raw_hit': bool(top2_raw_hit),
                    'top2_hit': bool(top2_hit),
                    'bert_text_available': bool(text_a.strip() != '' and text_b.strip() != ''),
                    'sentences_a_count': len(sentences_a),
                    'sentences_b_count': len(sentences_b),
                    'top_feature_pairs': sentence_pairs[:2]
                }
                results_by_pair_idx[int(it['pair_index'])]['feature_shapley_core_modality'] = feature_details
            if bert_sentence_model is not None:
                try:
                    bert_sentence_model.to('cpu')
                except Exception:
                    pass
            del bert_sentence_model
            del bert_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        pair_arr_mod = np.array([[int(it['drug_a_idx']), int(it['drug_b_idx'])] for it in group_items], dtype=np.int64)
        pair_tensor_mod = torch.tensor(pair_arr_mod, dtype=torch.long, device=local_device)
        target_classes_mod = [int(it['target_class']) for it in group_items]

        mask_values_cur = EXPLAIN_FEATURE_MASK_VALUES_BY_MODALITY.get(core_modality, [EXPLAIN_MASK_VALUE])
        if len(mask_values_cur) == 0:
            mask_values_cur = [EXPLAIN_MASK_VALUE]
        selected_mask_for_output = float(mask_values_cur[0])

        for mask_idx, feature_mask_value in enumerate(mask_values_cur):
            candidates_all = []
            for it in group_items:
                cands = build_core_feature_candidates_by_modality(
                    int(it['drug_a_idx']), int(it['drug_b_idx']), core_modality,
                    feats_1, feats_2, feats_4, dataset1_tails, dataset2_tails, dataset4_tails, dataset1_rel_map
                )
                candidates_all.append(cands)

            accum_all = [[[] for _ in cands] for cands in candidates_all]
            max_cand = max([len(c) for c in candidates_all]) if len(candidates_all) > 0 else 0
            for cand_idx in range(max_cand):
                active_idx = [gi for gi, cands in enumerate(candidates_all) if cand_idx < len(cands)]
                if len(active_idx) == 0:
                    continue
                for _ in range(EXPLAIN_MC_SAMPLES_FEATURE):
                    ops_t_by_pair = [[] for _ in range(len(group_items))]
                    ops_ts_by_pair = [[] for _ in range(len(group_items))]
                    for gi in active_idx:
                        cands = candidates_all[gi]
                        other_idx = [i for i in range(len(cands)) if i != cand_idx]
                        k = rng.randint(0, len(other_idx))
                        subset = rng.sample(other_idx, k) if k > 0 else []
                        ops_t = []
                        for oi in subset:
                            ops_t.extend(build_feature_ops(cands[oi]))
                        ops_ts = list(ops_t) + build_feature_ops(cands[cand_idx])
                        ops_t_by_pair[gi] = ops_t
                        ops_ts_by_pair[gi] = ops_ts

                    scores_t = apply_ops_and_score_batch_union(net, pair_tensor_mod, target_classes_mod, ops_t_by_pair, mask_val=float(feature_mask_value))
                    scores_ts = apply_ops_and_score_batch_union(net, pair_tensor_mod, target_classes_mod, ops_ts_by_pair, mask_val=float(feature_mask_value))
                    contrib = scores_t - scores_ts
                    for gi in active_idx:
                        accum_all[gi][cand_idx].append(float(contrib[gi]))

            mask_top1 = 0
            mask_top2 = 0
            for gi, it in enumerate(group_items):
                cands = candidates_all[gi]
                feat_scores = []
                for ci, cand in enumerate(cands):
                    vals = accum_all[gi][ci]
                    feat_scores.append({
                        'modality': cand['modality'],
                        'rule': cand['rule'],
                        'feature_a': cand['a_name'],
                        'feature_b': cand['b_name'],
                        'a_relations': cand.get('a_relations', []),
                        'b_relations': cand.get('b_relations', []),
                        'role_pairs': cand.get('role_pairs', []),
                        'shapley_value': float(np.mean(vals)) if len(vals) > 0 else 0.0
                    })
                feat_scores.sort(key=lambda x: x['shapley_value'], reverse=True)

                top1_feat = feat_scores[0] if len(feat_scores) >= 1 else None
                top2_feat = feat_scores[1] if len(feat_scores) >= 2 else None
                top1_hit = check_hit_logic(core_modality, top1_feat)
                top2_raw_hit = check_hit_logic(core_modality, top2_feat)
                top2_hit = bool(top2_raw_hit) if args.top2_mode == "strict" else bool(top1_hit or top2_raw_hit)
                if top1_hit:
                    mask_top1 += 1
                if top2_hit:
                    mask_top2 += 1

                # 仅用该模态的首个mask值写入最终结果与总体统计
                if mask_idx != 0:
                    continue

                if len(cands) >= 2:
                    cand_ge2_stats[core_modality] += 1
                if top2_raw_hit:
                    hit_stats_raw_top2[core_modality] += 1
                model1_rule_hits = None
                if core_modality == "Model1-Model1":
                    model1_rule_hits = {}
                    top1_rule_hits = set()
                    for rk in ['enzyme-enzyme', 'transporter-transporter', 'target-target']:
                        if check_model1_rule_hit(rk, top1_feat):
                            top1_rule_hits.add(rk)
                    for rk in ['enzyme-enzyme', 'transporter-transporter', 'target-target']:
                        rk_top1 = check_model1_rule_hit(rk, top1_feat)
                        rk_top2_raw = check_model1_rule_hit(rk, top2_feat)
                        if len(top1_rule_hits) > 0 and rk not in top1_rule_hits and args.top2_mode != "strict":
                            rk_top2_raw = False
                        if args.top2_mode == "strict":
                            rk_top2 = bool(rk_top2_raw)
                        else:
                            rk_top2 = bool(rk_top1 or rk_top2_raw)
                        model1_rule_hits[rk] = {'top1': bool(rk_top1), 'top2_raw': bool(rk_top2_raw), 'top2': bool(rk_top2)}
                        if rk_top1:
                            hit_stats_model1_rules[rk]['top1'] += 1
                        if rk_top2:
                            hit_stats_model1_rules[rk]['top2'] += 1

                if core_modality in hit_stats:
                    if top1_hit:
                        hit_stats[core_modality]['top1'] += 1
                    if top2_hit:
                        hit_stats[core_modality]['top2'] += 1
                if core_modality == "Model1-Model1" and top2_hit:
                    model1_top2_dedup_total += 1
                if top1_hit:
                    total_top1_hit += 1
                if top2_hit:
                    total_top2_hit += 1

                feature_details = {
                    'core_modality': core_modality,
                    'feature_mask_value': float(feature_mask_value),
                    'candidate_count': len(cands),
                    'top1': top1_feat,
                    'top2': top2_feat,
                    'top1_hit': bool(top1_hit),
                    'top2_raw_hit': bool(top2_raw_hit),
                    'top2_hit': bool(top2_hit),
                    'model1_rule_hits': model1_rule_hits,
                    'top_feature_pairs': feat_scores[:2]
                }
                results_by_pair_idx[int(it['pair_index'])]['feature_shapley_core_modality'] = feature_details

            print(f"[Explain][Stage2] {core_modality} 特征mask={float(feature_mask_value)} -> top1={mask_top1}/{len(group_items)}, top2={mask_top2}/{len(group_items)}")

    results = [results_by_pair_idx[k] for k in sorted(results_by_pair_idx.keys())]
    print(f"[Explain][Stage2] 各核心模态对内命中率 (top2_mode={args.top2_mode}):")
    for mod_name in ['Model1-Model1', 'Model2-Model1', 'Model4-Model1', 'BertMLP-BertMLP']:
        denom = int(stage1_modality_top1_hits.get(mod_name, 0))
        top1_cnt = int(hit_stats.get(mod_name, {}).get('top1', 0))
        top2_cnt = int(hit_stats.get(mod_name, {}).get('top2', 0))
        top2_raw_cnt = int(hit_stats_raw_top2.get(mod_name, 0))
        cand_ge2_cnt = int(cand_ge2_stats.get(mod_name, 0))
        top1_rate = (top1_cnt / denom) if denom > 0 else 0.0
        top2_rate = (top2_cnt / denom) if denom > 0 else 0.0
        print(f"  {mod_name}: top1={top1_cnt}/{denom} ({top1_rate:.4f}), top2={top2_cnt}/{denom} ({top2_rate:.4f}), top2_raw={top2_raw_cnt}/{denom}, cand_ge2={cand_ge2_cnt}/{denom}")
    model1_denom = int(stage1_modality_top1_hits.get('Model1-Model1', 0))
    print("  Model1-Model1 细分命中:")
    for rk in ['enzyme-enzyme', 'transporter-transporter', 'target-target']:
        rk_top1 = int(hit_stats_model1_rules[rk]['top1'])
        rk_top2 = int(hit_stats_model1_rules[rk]['top2'])
        rk_top1_rate = (rk_top1 / model1_denom) if model1_denom > 0 else 0.0
        rk_top2_rate = (rk_top2 / model1_denom) if model1_denom > 0 else 0.0
        print(f"    {rk}: top1={rk_top1}/{model1_denom} ({rk_top1_rate:.4f}), top2={rk_top2}/{model1_denom} ({rk_top2_rate:.4f})")
    dedup_rate = (model1_top2_dedup_total / model1_denom) if model1_denom > 0 else 0.0
    print(f"    top2_dedup_total={model1_top2_dedup_total}/{model1_denom} ({dedup_rate:.4f})")

    summary = {
        'drug_id': DRUG_ID,
        'drug_name': DRUG_NAME,
        'fold_id': int(fold_id),
        'epoches': int(args.epoches),
        'mc_samples_modality': int(EXPLAIN_MC_SAMPLES_MODALITY),
        'mc_samples_feature': int(EXPLAIN_MC_SAMPLES_FEATURE),
        'bert_text_source_file': args.bert_text_source_file,
        'bert_model_path': args.bert_model_path,
        'top2_mode': args.top2_mode,
        'bert_semantic_tags': BERT_SEMANTIC_TAGS,
        'stage1_mask_values': [float(x) for x in EXPLAIN_STAGE1_MASK_VALUES],
        'stage2_feature_mask_values_by_modality': {
            k: [float(x) for x in v] for k, v in EXPLAIN_FEATURE_MASK_VALUES_BY_MODALITY.items()
        },
        'stage1_selected_mask_value_for_stage2': float(selected_mask_value),
        'stage1_scan_results': stage1_scan_results,
        'stage2_top2_raw_hits': hit_stats_raw_top2,
        'stage2_cand_ge2_counts': cand_ge2_stats,
        'total_pairs': int(len(results)),
        'stage1_modality_top1_hits': stage1_modality_top1_hits,
        'hit_stats_by_modality': hit_stats,
        'hit_stats_model1_rules': hit_stats_model1_rules,
        'model1_top2_dedup_total': int(model1_top2_dedup_total),
        'total_top1_hit': int(total_top1_hit),
        'total_top2_hit': int(total_top2_hit),
        'pairs': results
    }

    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Explain] Shapley归因已保存: {save_file}")


def train_gpu(train_x,train_y,test_x,test_y,net,test_adj, dict1, tokenizer, drug_emb, real_edge_1, real_edge_2, real_edge_4, feats_1, feats_2, feats_4, main_l, aux_l, aux_l2, dataset1_tails, dataset2_tails, dataset4_tails, fold_id):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    max_test_output = torch.zeros((0,47),dtype=torch.float).to(device)
    real_edge_1 = real_edge_1.to('cuda:0')
    real_edge_2 = real_edge_2.to('cuda:0')
    real_edge_4 = real_edge_4.to('cuda:0')
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
            

            # l = 1 * loss_function(final_embed, train_label) + \
            #     0.06 * loss_function(z1, real_edge_1) + \
            #     0.06 * loss_function(z2, real_edge_2) + \
            #     0.06 * loss_function(z4, real_edge_4)
            main_loss = loss_function(final_embed, train_label)
            recon1 = loss_function(z1, real_edge_1)
            recon2 = loss_function(z2, real_edge_2)
            recon3 = loss_function(z4, real_edge_4)
            gae1_loss = loss_function(emb1, train_label)

            
            l = main_l * main_loss + \
                aux_l * recon1 + \
                aux_l * recon2 + \
                aux_l * recon3 + \
                aux_l * gae1_loss
            
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
    run_explainability_after_training(
        net=net,
        dict1=dict1,
        feats_1=feats_1,
        feats_2=feats_2,
        feats_4=feats_4,
        dataset1_tails=dataset1_tails,
        dataset2_tails=dataset2_tails,
        dataset4_tails=dataset4_tails,
        fold_id=fold_id
    )
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
    max_test_output = torch.zeros((0,47),dtype=torch.float)
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
    sim_matrix4 = np.zeros((1621, 1621), dtype=float)
    for i in range(1621):
        for j in range(1621):
            for k in range(20):
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

def main():
    # conn = sqlite3.connect(path_prefix + "/dataset/event.db")
    # df_drug = pd.read_sql('select * from drug;', conn)
    # extraction = pd.read_sql('select * from extraction;', conn)
    # mechanism = extraction['mechanism']
    # action = extraction['action']
    # drugA = extraction['drugA']
    # drugB = extraction['drugB']
    ddi_file_path = '/home/jy/桌面/MSE-ENN/Dataset2/final.csv'
    drug_file_path = '/home/jy/桌面/MSE-ENN/Dataset2/drug_smiles.csv'
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
    dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1,1)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1,2)
    # dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1,3)
    dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1,4)
    dataset1_tails = build_tail_map(1)
    dataset2_tails = build_tail_map(2)
    dataset4_tails = build_tail_map(4)
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

    feature_matrix_origin = torch.randn((1621+825, 256), dtype=float)
    # 药物与药物子结构的节点与特征
    feats = {}

    feats_1 = np.zeros((1621, dataset1_tail_len), dtype=float)
    feats_2 = np.zeros((1621, dataset2_tail_len), dtype=float)
    feats_4 = np.zeros((1621, dataset4_tail_len), dtype=float)

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
                    edge_index_1[1].append(1621 + j[0])
                    real_edge_1.append(j[1])
                    num_classes_1.append(j[1])
                elif p == 1:
                    feats_2[i][j[0]] = 1
                    edge_index_2[0].append(i)
                    edge_index_2[1].append(1621 + j[0])
                    real_edge_2.append(1)
                elif p == 2:
                    feats_4[i][j[0]] = j[1]
                    edge_index_4[0].append(i)
                    edge_index_4[1].append(1621 + j[0])
                    real_edge_4.append(j[1])
                    num_classes_4.append(j[1])

    edge_index_1 = torch.tensor(edge_index_1)
    edge_index_2 = torch.tensor(edge_index_2)
    edge_index_4 = torch.tensor(edge_index_4)

    num_classes_1 = set(num_classes_1)
    num_classes_4 = set(num_classes_4)

    # from torch_geometric.data import HeteroData
    # data = HeteroData()

    # data['drug'].x = torch.randn(1621, 256)
    # data['chemical'].x = torch.randn(825, 256)

    # data['drug', 'interacts', 'chemical'].edge_index = torch.tensor(edge_index)
    # data['drug', 'interacts', 'chemical'].edge_attr = torch.tensor(edge_weight_tensor).view(-1, 1)

    # print(data)
    # feats_1 = torch.tensor(feats_1)

    # drug_idx, chemical_idx = torch.nonzero(feats_1, as_tuple=True)
    # chemical_idx += 1621
    # edge_index = torch.stack([drug_idx, chemical_idx], dim=0)
    # edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # from torch_geometric.datasets import Planetoid
    # import torch_geometric.transforms as T
    # from torch_geometric.utils import train_test_split_edges
    # dataset = Planetoid("/Users/charly/Downloads/MSE-ENN-FENN-main/Code and Datasets/data/data/", "CiteSeer", transform=T.NormalizeFeatures())
    # print(dataset.data)

    # data = dataset[0]
    # data.train_mask = data.val_mask = data.test_mask = data.y = None
    # data = train_test_split_edges(data)


    # feature_matrix3 = np.zeros((1621, 1621), dtype=float)
    feature_matrix1 = np.zeros((1621, dataset1_tail_len), dtype=float)
    feature_matrix2 = np.zeros((1621, dataset2_tail_len), dtype=float)
    feature_matrix3 = np.zeros((1621, dataset4_tail_len), dtype=float)
    feature_matrix4 = np.zeros((1621, 1621), dtype=float)

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
    y_score = np.zeros((0, 47), dtype=float)
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
    drug_emb = torch.load('/home/jy/桌面/MSE-ENN/Dataset2/drug_desc_udever_embeds.pt')
    # torch.save(drug_emb, '/home/jy/桌面/MSE-ENN/result4/100_baike_duiqi_tip_newmodel/drug_emb.pt')
    for main_l in [1]:
        for l in [0.06]:
            for l2 in [0.0000000000000001]:
                for cross_ver in range(0, 5):

                    net = nn.Sequential(Model1(args, dataset1_tail_len, feats_1, edge_index_1, 512, args.embedding_num, args.embedding_num, args.embedding_num, len(num_classes_1), real_edge_1),
                                        Model2(args, dataset2_tail_len, feats_2, edge_index_2, 512, args.embedding_num, args.embedding_num, args.embedding_num, 2, real_edge_2),
                                        Model4(args, dataset4_tail_len, feats_4, edge_index_4, 512, args.embedding_num, args.embedding_num, args.embedding_num, len(num_classes_4), real_edge_4),
                                        BertMLP('..'),
                                        FusionLayer1(args))

                    # bert_model = BertMLM(r"../model/base-uncased")
                    
                    # net = nn.Sequential(bert_model, FusionLayer(args))
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
                                                                                            net,test_adj[cross_ver], dict1, tokenizer, drug_emb, real_edge_1, real_edge_2, real_edge_4, feats_1, feats_2, feats_4, main_l, l, l2, dataset1_tails, dataset2_tails, dataset4_tails, cross_ver)
                    train_sum += train_acc
                    test_sum += test_acc
                    test_output = test_output.to('cpu')
                    pred_type = torch.argmax(test_output, dim=1).numpy()
                    y_pred = np.hstack((y_pred, pred_type))
                    y_score = np.row_stack((y_score, test_output))
                    y_true = np.hstack((y_true, test_y))
                    print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
                            cross_ver, test_loss, test_acc, train_loss, train_acc))

                result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
                save_result(path_prefix + "/result_Dataset2_task2/loss_mlp_" + str(main_l) + '_' + str(l) + '_' + str(l2) + '_', "all", result_all)
                save_result(path_prefix + "/result_Dataset2_task2/loss_mlp_" + str(main_l) + '_' + str(l) + '_', "each", result_eve)
                print('%d-fold validation: avg train acc  %f, avg test acc %f' % (5, train_sum / 5, test_sum / 5))
    return

if __name__ == '__main__':
    main()
