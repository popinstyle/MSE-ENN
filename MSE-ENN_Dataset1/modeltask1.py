import torch
import torch.nn as nn
import torch.nn.functional as F
# from keras import backend as K
from collections import defaultdict
import numpy as np
import copy
import os
from tqdm import tqdm
from torch_geometric.nn import VGAE
from transformers import AutoTokenizer, BloomModel

from torch_geometric.nn import GCNConv,RGCNConv
import math

    
# from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
# from transformers import AutoConfig, BertModel

EPS = 1e-15
MAX_LOGSTD = 10

device = 'cuda:0'

path_prefix = r'C:/Users/Administrator/Desktop/code/Code and Datasets/'

def rm_forward(key_str, j):
    key_list = key_str.split('.')
    key_list = key_list[j:]
    return '.'.join(key_list)

class Model1(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge):
        super(Model1, self).__init__()
        self.inx = '1'
        self.autoencoder = GCNEncoder(args, tail_len, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge, self.inx)

    def forward(self, args):
        idx, feats = args
        drug_f, z = self.autoencoder(feats)
        return drug_f, z, idx, feats
    
class Model2(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge):
        super(Model2, self).__init__()
        self.inx = '2'
        self.autoencoder = GCNEncoder(args, tail_len, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge, self.inx)

    def forward(self, arguments):
        gae1_embedding, gae1_z, idx, feats = arguments
        drug_f, z = self.autoencoder(feats)

        return drug_f, z, gae1_embedding, gae1_z, idx, feats
    
class Model4(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge):
        super(Model4, self).__init__()
        self.inx = '4'
        self.autoencoder = GCNEncoder(args, tail_len, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge, self.inx)

    def forward(self, arguments):
        
        gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx, feats = arguments
        drug_f, z = self.autoencoder(feats)

        return drug_f, z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx, feats
    
class BertMLP(nn.Module):
    def __init__(self, embedding):
        super(BertMLP, self).__init__()
        self.bert_embedding = embedding
        self.drug_emb = torch.load('/home/jy/桌面/MSE-ENN/dataset5/drug_emb.pt').to('cuda:0')
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 384)
        )
        
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
    
    def forward(self, args):
        # 接收来自Model4的输出，并透传feats用于Fusion层的门控
        if isinstance(args, (list, tuple)) and len(args) == 8:
            gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx, feats = args
        else:
            # 兼容旧调用（无feats透传）
            gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = args
            feats = None

        # 如果feats中有feats_bert，使用它（支持掩码）；否则使用预加载的drug_emb
        if feats is not None and isinstance(feats, dict) and 'feats_bert' in feats:
            x = torch.tensor(feats['feats_bert']).to('cuda:0').float()
        else:
            x = self.drug_emb

        return self.fullConnectionLayer(x), gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx, feats

class FusionLayer1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(args.embedding_num*8, args.embedding_num*4),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num*4),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num*4,args.embedding_num*2),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num*2),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num*2,4))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        # 兼容两种调用：含feats与不含feats
        if isinstance(arguments, (list, tuple)) and len(arguments) == 9:
            llm_embedding, gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx, feats = arguments
        else:
            llm_embedding, gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = arguments
            feats = None
        drugA = idx[:, 0]  # 取第0列，起点
        drugB = idx[:, 1]  # 取第1列，终点

        # 先取出未遮罩向量
        a_g1 = gae1_embedding[drugA]
        a_g2 = gae2_embedding[drugA]
        a_g4 = gae4_embedding[drugA]
        a_llm = llm_embedding[drugA]
        b_g1 = gae1_embedding[drugB]
        b_g2 = gae2_embedding[drugB]
        b_g4 = gae4_embedding[drugB]
        b_llm = llm_embedding[drugB]

        # 按需执行“跨/内模态”的单侧遮罩：仅对(a的指定模态)与(b的指定模态)置零
        if isinstance(feats, dict) and ('pair_cross_mask' in feats):
            pcm = feats['pair_cross_mask'] or {}
            a_mod = pcm.get('a', None)
            b_mod = pcm.get('b', None)
            if a_mod == 'dataset1': a_g1 = torch.zeros_like(a_g1)
            elif a_mod == 'dataset2': a_g2 = torch.zeros_like(a_g2)
            elif a_mod == 'dataset4': a_g4 = torch.zeros_like(a_g4)
            elif a_mod == 'bert':    a_llm = torch.zeros_like(a_llm)
            if b_mod == 'dataset1': b_g1 = torch.zeros_like(b_g1)
            elif b_mod == 'dataset2': b_g2 = torch.zeros_like(b_g2)
            elif b_mod == 'dataset4': b_g4 = torch.zeros_like(b_g4)
            elif b_mod == 'bert':    b_llm = torch.zeros_like(b_llm)

        emb1 = torch.cat([a_g1, b_g1], 1).float()
        emb2 = torch.cat([a_g2, b_g2], 1).float()
        emb4 = torch.cat([a_g4, b_g4], 1).float()

        Embedding = torch.cat([
                               a_g1, a_g2, a_g4, a_llm,
                               b_g1, b_g2, b_g4, b_llm
                               ], 1).float()
        
        return self.fullConnectionLayer(Embedding), gae4_z, gae2_z, gae1_z, emb4, emb2, emb1


class ModalityCrossAttentionFusion(nn.Module):
    def __init__(self, args, d_model=256, n_heads=4, d_ff=512, dropout=0.1):
        super(ModalityCrossAttentionFusion, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        # 4模态 * 2药物 = 8 token
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 融合后接前馈层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        # 最终分类或DDI预测头
        self.out_proj = nn.Linear(d_model, 4)  # 或num_classes
    
    def forward(self, arguments):
        """
        mod_a_list: List[Tensor], 长度4，每个Tensor shape [batch, d_model]
        mod_b_list: List[Tensor], 长度4，每个Tensor shape [batch, d_model]
        返回: DDI预测 logits [batch, 1]
        """
        llm_embedding, gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = arguments
        # 1024*2
        drugA = idx[:, 0]  # 取第0列，起点
        drugB = idx[:, 1]  # 取第1列，终点
        mod_a_list = [
            llm_embedding[drugA],    # [batch, d_model]
            gae4_embedding[drugA],
            gae2_embedding[drugA],
            gae1_embedding[drugA],
        ]
        mod_b_list = [
            llm_embedding[drugB],
            gae4_embedding[drugB],
            gae2_embedding[drugB],
            gae1_embedding[drugB],
        ]
        x = torch.stack(mod_a_list + mod_b_list, dim=1)  # [batch, 8, d_model]
        B, N, D = x.shape  # N=8, D=d_model

        # QKV变换
        Q = self.query_proj(x)  # [B, 8, D]
        K = self.key_proj(x)
        V = self.value_proj(x)

        # 分多头
        def split_heads(tensor):
            return tensor.reshape(B, N, self.n_heads, self.head_dim).transpose(1,2)
        
        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Attention: [B, n_heads, 8, 8]
        attn_weights = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 聚合
        attn_out = attn_weights @ V  # [B, n_heads, 8, head_dim]
        attn_out = attn_out.transpose(1,2).contiguous().reshape(B, N, D)  # [B, 8, d_model]

        # 池化聚合成1个融合向量
        fusion = attn_out.mean(dim=1)  # [B, d_model]
        fusion = self.ffn(fusion)      # 前馈层
        llm_emb = torch.cat([gae1_embedding[drugA],gae2_embedding[drugA]], 1).float()

        logits = self.out_proj(fusion) # [B, 1] 或 [B, num_classes]
        return logits, llm_emb, gae4_z, gae2_z, gae1_z

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class feature_encoder(torch.nn.Module):  # twin network
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge):
        super(feature_encoder, self).__init__()

        # self.layers = torch.nn.ModuleList([EncoderLayer(vector_size, n_heads) for _ in range(n_layers)])
        self.layer = nn.Linear(tail_len, in_channels)
        self.AN = torch.nn.LayerNorm(in_channels)

        # self.l1 = torch.nn.Linear(512, 512)
        self.bn1 = torch.nn.BatchNorm1d(in_channels)

        self.dr = torch.nn.Dropout(args.dropout)

        self.ac = torch.nn.ReLU()

    def forward(self, X):
        # for layer in self.layers:
        #     X = layer(X)

        X1 = self.AN(self.layer(X))
        X2 = self.dr(self.bn1(self.ac(X1)))

        return X2
    
class GCNEncoder(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge, inx):
        super(GCNEncoder, self).__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量

        self.drug_features = torch.tensor(feats).to('cuda:0')
        self.chemical_features = torch.tensor(feats.T).to('cuda:0')
        self.edge_index = edge_index.to('cuda:0')
        self.real_edge = real_edge.to('cuda:0')
        
        self.drug_projector = feature_encoder(tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge)
        self.chemical_projector = feature_encoder(1409, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge)
        
        # self.drug_projector = nn.Linear(tail_len, 512)
        # self.chemical_projector = nn.Linear(1409, 512)
        self.relu = nn.ReLU()

        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        # self.conv3 = GCNConv(2 * out_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        

        if hidden_dim is not None:
            self.proj_z = True
            self.lin = nn.Linear(hidden_dim, output_dim)
        
        self.edge_classifier = nn.Linear(2 * output_dim, num_classes)
        self.inx = inx

    def forward(self, feats):

        drug_features = torch.tensor(feats['feats_' + self.inx]).to('cuda:0')
        chemical_features = torch.tensor(feats['feats_' + self.inx].T).to('cuda:0')

        drug_x = self.drug_projector(drug_features.float())
        chemical_x = self.chemical_projector(chemical_features.float())
        # drug_x = F.normalize(drug_x, p=2, dim=1)
        # chemical_x = F.normalize(chemical_x, p=2, dim=1)
        feat_final = torch.cat([drug_x, chemical_x], dim=0)

        x1 = self.conv1(feat_final, self.edge_index).relu()
        # x1 = self.conv3(x1, self.edge_index).relu()
        drug_f = self.conv2(x1, self.edge_index)

        z = self.lin(drug_f) if self.proj_z else z

        # z = z.detach()        
        # 重构的edge权重
        # value = (z[self.edge_index[0]] * z[self.edge_index[1]]).sum(dim=1)
        src_emb = z[self.edge_index[0]]  # 起点节点的embedding
        dst_emb = z[self.edge_index[1]]  # 终点节点的embedding
        edge_feat = torch.cat([src_emb, dst_emb], dim=1)
        return drug_f, self.edge_classifier(edge_feat)

class VGCNEncoder(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes):
        super(VGCNEncoder, self).__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量

        self.drug_features = torch.tensor(feats).to('cuda:0')
        self.chemical_features = torch.tensor(feats.T).to('cuda:0')
        self.edge_index = edge_index.to('cuda:0')
        
        self.drug_projector = nn.Linear(tail_len, 256)
        self.chemical_projector = nn.Linear(1409, 256)
        self.relu = nn.ReLU()
        
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

        if hidden_dim is not None:
            self.proj_z = True
            self.lin = nn.Linear(hidden_dim, output_dim)
        
        self.edge_classifier = nn.Linear(2 * output_dim, num_classes)
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
    
    def kl_loss(self, mu, logstd):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def forward(self):
        
        drug_x = self.relu(self.drug_projector(self.drug_features.float()))
        chemical_x = self.relu(self.chemical_projector(self.chemical_features.float()))
        drug_x = F.normalize(drug_x, p=2, dim=1)
        chemical_x = F.normalize(chemical_x, p=2, dim=1)
        feat_final = torch.cat([drug_x, chemical_x], dim=0)

        x1 = self.conv1(feat_final, self.edge_index).relu()
        # drug_f = self.conv2(x1, self.edge_index)

        self.__mu__ = self.conv_mu(x1, self.edge_index)
        self.__logstd__ = self.conv_logstd(x1, self.edge_index)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)

        drug_f = self.reparametrize(self.__mu__, self.__logstd__)

        z = self.lin(drug_f) if self.proj_z else z
        # # 重构的edge权重
        # # value = (z[self.edge_index[0]] * z[self.edge_index[1]]).sum(dim=1)
        src_emb = z[self.edge_index[0]]  # 起点节点的embedding
        dst_emb = z[self.edge_index[1]]  # 终点节点的embedding
        edge_feat = torch.cat([src_emb, dst_emb], dim=1)
        return drug_f, edge_feat, self.kl_loss(self.__mu__, self.__logstd__)
#   {
#   "epoch": 1,
#   "total_pairs": 151,
#   "modality_pair_max_counts": {
#     "a_dataset1|b_dataset1": 17,
#     "a_dataset1|b_bert": 0,
#     "a_dataset2|b_dataset1": 84,
#     "a_dataset2|b_bert": 0,
#     "a_dataset4|b_dataset1": 0,
#     "a_dataset4|b_bert": 0,
#     "a_bert|b_dataset1": 0,
#     "a_bert|b_bert": 50
#   }
# }