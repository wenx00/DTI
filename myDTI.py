"""
Note: this model is unfinished!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv
# from dgl.contrib.sampling.sampler import *
import pandas as pd
import numpy as np
import dgl
import dgl.function as fn
import json
import pickle
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from utils import *

# ===================================================
# # Load Data
Drug_Drug_Interactions = load_csv_file("./assets/DDI_dataset.csv")
Drug_Protein_Interactions = load_csv_file("./assets/DPI_dataset.csv")
Protein_Protein_Interactions = load_csv_file("./assets/PPI_dataset.csv")
# Protein_Drug_Interactions = load_csv_file('PDI_dataset.csv')

# Create Graph
g = dgl.heterograph({
    ('drug', 'DPI', 'protein'): Drug_Protein_Interactions,
    # ('protein', 'PDI', 'drug'): Protein_Drug_Interactions,
    ('drug', 'DDI', 'drug'): Drug_Drug_Interactions,
    ('protein', 'PPI', 'protein'): Protein_Protein_Interactions
})
with open('./assets/g.pkl', 'wb') as f:
    pickle.dump(g, f)
# Check Graph
g.number_of_nodes('drug')
g.number_of_edges('PPI')

drug_features = torch.empty((g.number_of_nodes('drug'), 1024))
protein_features = torch.empty((g.number_of_nodes('protein'), 1024))
nn.init.xavier_uniform_(drug_features, gain=nn.init.calculate_gain('relu'))
nn.init.xavier_uniform_(protein_features, gain=nn.init.calculate_gain('relu'))
node_features = {
    'drug': drug_features,
    'protein': protein_features
}
g.ndata['feat'] = node_features

with open('./assets/g_feat.pkl', 'wb') as f:
    pickle.dump(g, f)
# ===================================================
input("Done, plz terminate it!")


with open('g.pkl', 'rb') as f:
    g = pickle.load(f)


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype: nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get paper logits
        return h_dict['drug']


# generate train/val/test split


model = HeteroRGCN(g, 10, 10, 3)

opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


class HisDTI(nn.Module):
    def __init__(self, g, gnn_layers, in_dim, hidden_dimensions, num_rels, activations, feat_drop, num_bases=-1):
        super(HisDTI, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.hidden_dimensions = hidden_dimensions
        self.num_channels = hidden_dimensions[-1]
        self.num_rels = num_rels
        self.feat_drop = feat_drop
        self.num_bases = num_bases
        self.activations = activations
        self.gnn_layers = gnn_layers
        # create RGCN layers
        self.build_model()

    def set_g(self, g):
        self.g = g

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for i in range(self.gnn_layers - 2):
            h2h = self.build_hidden_layer(i)
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    def build_input_layer(self):
        print('Building an INPUT  layer of {}x{} (rels:{})'.format(self.in_dim, self.hidden_dimensions[0],
                                                                   self.num_rels))
        return RelGraphConv(self.in_dim, self.hidden_dimensions[0], self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.activations[0])

    def build_hidden_layer(self, i):
        print('Building an HIDDEN  layer of {}x{}'.format(self.hidden_dimensions[i], self.hidden_dimensions[i + 1]))
        return RelGraphConv(self.hidden_dimensions[i], self.hidden_dimensions[i + 1], self.num_rels,
                            regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.activations[i + 1])

    def build_output_layer(self):
        print('Building an OUTPUT  layer of {}x{}'.format(self.hidden_dimensions[-2], self.hidden_dimensions[-1]))
        return RelGraphConv(self.hidden_dimensions[-2], self.hidden_dimensions[-1], self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.activations[-1])

    def forward(self, features, etypes):
        h = features
        self.g.edata['norm'] = self.g.edata['norm'].to(device=features.device)

        for layer in self.layers:
            h = layer(self.g, h, etypes)
        return h


class MyDTI(nn.Module):
    def __init__(self, g, in_feat, out_feat, num_rels, regularizer):
        super(MyDTI, self).__init__()
        self.g = g
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.layers = nn.ModuleList()
        self.build_model()

    def build_model(self):
        g = self.g
        self.layers.append(RelGraphConv(self.in_feat, self.out_feat, self.num_rels, self.regularizer))
        self.layers.append(RelGraphConv(self.in_feat, self.out_feat, self.num_rels, self.regularizer))
        self.layers.append(RelGraphConv(self.in_feat, self.out_feat, self.num_rels, self.regularizer))

    def forward(self, features, etypes):
        # self.g.edata['norm'] = self.g.edata['norm'].to(device=features.device)
        h = features
        for layer in self.layers:
            h = layer(self.g, h, etypes)
        return h


mmm = MyDTI(g, 8, 8, 4, 'bdd')

print(mmm)


def sample_blocks(self, seeds):
    blocks = []
    seeds = {self.category: torch.tensor(seeds).long()}
    cur = seeds
    for fanout in self.fanouts:
        if fanout is None:
            frontier = dgl.in_subgraph(self.g, cur)
        else:
            frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
        block = dgl.to_block(frontier, cur)
        cur = {}
        for ntype in block.srctypes:
            cur[ntype] = block.srcnodes[ntype].data[dgl.NID]
        blocks.insert(0, block)
    return seeds, blocks

# for epoch in range(100):
#     logits = model(g)
#     # The loss is computed only for labeled nodes.
#     loss = F.cross_entropy(logits[train_idx], labels[train_idx])
#
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#
#     if epoch % 5 == 0:
#         print(epoch)
#
