import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import time
import dgl.nn.pytorch as dglnn
import pickle
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score
import tqdm

with open('./assets/g_feature.pkl', 'rb') as f:
    g = pickle.load(f)


class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = torch.empty(3, 2048)
        nn.init.xavier_uniform_(self.params, gain=nn.init.calculate_gain('relu'))
        # self.rels = nn.ParameterDict({
        #     'DPI': nn.Parameter(torch.tensor(self.params[0], requires_grad=True)),
        #     'DDI': nn.Parameter(torch.tensor(self.params[1], requires_grad=True)),
        #     'PPI': nn.Parameter(torch.tensor(self.params[2], requires_grad=True))
        # })
        # self.rels = nn.ParameterDict({
        #     'DPI': nn.Parameter(self.params[0].clone().detach().requires_grad_(True)),
        #     'DDI': nn.Parameter(self.params[1].clone().detach().requires_grad_(True)),
        #     'PPI': nn.Parameter(self.params[2].clone().detach().requires_grad_(True)),
        # })
        self.rels = nn.ParameterDict({
            'DPI': nn.Parameter(torch.ones_like(self.params[0], requires_grad=False)),
            'DDI': nn.Parameter(torch.ones_like(self.params[1], requires_grad=False)),
            'PPI': nn.Parameter(torch.ones_like(self.params[2], requires_grad=False))
        })

    def edge_func_ddi(self, edges):
        # todo: what is this # distmult
        head = edges.src['x']
        tail = edges.dst['x']
        score = head * self.rels['DDI'] * tail
        return {'score': torch.sum(score, dim=-1)}

    def edge_func_dpi(self, edges):
        # todo: what is this # distmult
        head = edges.src['x']
        tail = edges.dst['x']
        score = head * self.rels['DPI'] * tail
        return {'score': torch.sum(score, dim=-1)}

    def edge_func_ppi(self, edges):
        # todo: what is this # distmult
        head = edges.src['x']
        tail = edges.dst['x']
        score = head * self.rels['PPI'] * tail
        return {'score': torch.sum(score, dim=-1)}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(self.edge_func_ddi, etype=('drug', 'DDI', 'drug'))
            edge_subgraph.apply_edges(self.edge_func_dpi, etype=('drug', 'DPI', 'protein'))
            edge_subgraph.apply_edges(self.edge_func_dpi, etype=('protein', 'PPI', 'protein'))
            return edge_subgraph.edata['score']  # todo: what is this


class Model(nn.Module):
    def __init__(self, in_features, out_features, rel_names):
        super().__init__()
        self.layer = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_features, out_features, norm='right', activation=F.relu)
            for rel in rel_names
        })
        self.pred = ScorePredictor()

    def forward(self, pos_g, neg_g, blocks, x):
        x = self.layer(blocks[0], x)
        return self.pred(pos_g, x), self.pred(neg_g, x)

def compute_loss(pos_score, neg_score):
    # positive_logits # todo: what is this
    positive_logits = torch.cat(
        (pos_score[('drug', 'DPI', 'protein')],
         pos_score[('drug', 'DDI', 'drug')],
         pos_score[('protein', 'PPI', 'protein')]
         ), dim=0).cuda()
    # negative_logits
    negative_logits = torch.cat(
        (neg_score[('drug', 'DPI', 'protein')],
         neg_score[('drug', 'DDI', 'drug')],
         neg_score[('protein', 'PPI', 'protein')]
         ), dim=0).cuda()
    # loss function
    loss_func = nn.MSELoss()  # torch.nn.BCEWithLogitsLoss
    loss_func.cuda()
    # positive_score
    positive_score = loss_func(positive_logits, torch.ones_like(positive_logits))
    # negative_score
    negative_score = loss_func(negative_logits, torch.zeros_like(negative_logits))
    return torch.mean(torch.cat((positive_score.view(1), negative_score.view(1)), dim=0)), \
           positive_score, negative_score


feature_size = 2048

with open('./assets/g_feature.pkl', 'rb') as f:
    g = pickle.load(f)

rel_names = ['DDI', 'DPI', 'PPI']
checkpoint = torch.load('./assets/model_all_loss_0.966.pt')
model = Model(feature_size, feature_size, rel_names)
model.load_state_dict(checkpoint['model_state_dict'])

def score(drug, protein):
    aaa = torch.matmul(g.ndata['feat']['drug'][drug], model.layer.mods.DPI.weight)
    aaa = aaa + model.layer.mods.DPI.bias
    aaa=F.relu(aaa)
    bbb = torch.matmul(g.ndata['feat']['protein'][protein], model.layer.mods.DPI.weight)
    bbb = F.relu(bbb)
    bbb = bbb + model.layer.mods.DPI.bias
    s = torch.sum(aaa * bbb * model.pred.rels.DPI, dim=-1)
    print(s)