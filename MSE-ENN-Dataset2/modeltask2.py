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

device = 'cuda:1'

path_prefix = r'C:/Users/Administrator/Desktop/code/Code and Datasets/'

def rm_forward(key_str, j):
    key_list = key_str.split('.')
    key_list = key_list[j:]
    return '.'.join(key_list)

class Model1(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge):
        super(Model1, self).__init__()
        self.autoencoder = GCNEncoder(args, tail_len, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge)

    def forward(self, idx):
        drug_f, z = self.autoencoder()
        return drug_f, z, idx
    
class Model2(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge):
        super(Model2, self).__init__()
        self.autoencoder = GCNEncoder(args, tail_len, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge)

    def forward(self, arguments):
        gae1_embedding, gae1_z, idx = arguments
        drug_f, z = self.autoencoder()

        return drug_f, z, gae1_embedding, gae1_z, idx
    
class Model4(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge):
        super(Model4, self).__init__()
        self.autoencoder = GCNEncoder(args, tail_len, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge)

    def forward(self, arguments):
        
        gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = arguments
        drug_f, z = self.autoencoder()

        return drug_f, z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx
    
class BertMLP(nn.Module):
    def __init__(self, embedding):
        super(BertMLP, self).__init__()
        self.bert_embedding = embedding
        self.drug_emb = torch.load('/home/jy/桌面/MKG/dataset3/drug_emb_dataset3.pt').to('cuda:1')
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
        gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = args

        x = self.drug_emb

        return self.fullConnectionLayer(x), gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx

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
            nn.Linear(args.embedding_num*2,86))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        llm_embedding, gae4_embedding, gae4_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = arguments
        drugA = idx[:, 0]  # 取第0列，起点
        drugB = idx[:, 1]  # 取第1列，终点

        emb1 = torch.cat([gae1_embedding[drugA], gae1_embedding[drugB]], 1).float()
        emb2 = torch.cat([gae2_embedding[drugA], gae2_embedding[drugB]], 1).float()
        emb4 = torch.cat([gae4_embedding[drugA], gae4_embedding[drugB]], 1).float()

        Embedding = torch.cat([gae1_embedding[drugA], gae2_embedding[drugA], gae4_embedding[drugA], llm_embedding[drugA], \
                               gae1_embedding[drugB], gae2_embedding[drugB], gae4_embedding[drugB], llm_embedding[drugB]], 1).float()
        
        return self.fullConnectionLayer(Embedding), emb1, emb2, emb4, gae4_z, gae2_z, gae1_z


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
        self.out_proj = nn.Linear(d_model, 86)  # 或num_classes
    
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
        self.layer = nn.Linear(tail_len, 512)
        self.AN = torch.nn.LayerNorm(512)

        # self.l1 = torch.nn.Linear(512, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)

        self.dr = torch.nn.Dropout(args.dropout)

        self.ac = torch.nn.ReLU()

    def forward(self, X):
        # for layer in self.layers:
        #     X = layer(X)

        X1 = self.AN(self.layer(X))
        X2 = self.dr(self.bn1(self.ac(X1)))

        return X2
    
class GCNEncoder(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge):
        super(GCNEncoder, self).__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量

        self.drug_features = torch.tensor(feats).to('cuda:1')
        self.chemical_features = torch.tensor(feats.T).to('cuda:1')
        self.edge_index = edge_index.to('cuda:1')
        self.real_edge = real_edge.to('cuda:1')
        
        self.drug_projector = feature_encoder(tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge)
        self.chemical_projector = feature_encoder(1710, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge)
        
        # self.drug_projector = nn.Linear(tail_len, 512)
        # self.chemical_projector = nn.Linear(1710, 512)
        self.relu = nn.ReLU()

        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        # self.conv3 = GCNConv(2 * out_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        

        if hidden_dim is not None:
            self.proj_z = True
            self.lin = nn.Linear(hidden_dim, output_dim)
        
        self.edge_classifier = nn.Linear(2 * output_dim, num_classes)

    def forward(self):

        drug_x = self.drug_projector(self.drug_features.float())
        chemical_x = self.chemical_projector(self.chemical_features.float())
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

        self.drug_features = torch.tensor(feats).to('cuda:1')
        self.chemical_features = torch.tensor(feats.T).to('cuda:1')
        self.edge_index = edge_index.to('cuda:1')
        
        self.drug_projector = nn.Linear(tail_len, 256)
        self.chemical_projector = nn.Linear(1710, 256)
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
  