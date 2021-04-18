import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl
import time
import dgl.nn.pytorch as dglnn
import pickle
import itertools
import numpy as np
import tqdm

with open('./assets/g_feat.pkl', 'rb') as f:
    g = pickle.load(f)


rel_names = ['DDI', 'DPI', 'PPI']

class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class Model(nn.Module):
    def __init__(self, in_features, out_features, rel_names):
        super().__init__()
        self.layer = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_features, out_features, norm='right')
            for rel in rel_names
        })
        self.pred = ScorePredictor()

    def forward(self, pos_g, neg_g, blocks, x):
        x = self.layer(blocks[0], x)
        return self.pred(pos_g, x), self.pred(neg_g, x)


model = Model(1024, 1024, rel_names)
model = model.cuda()
opt = torch.optim.Adam(model.parameters())

model = Model(1024, 1024, rel_names)
model.load_state_dict(torch.load('./assets/model.pt'))
model.eval()
print(model.parameters())


model.eval()
valid_eid = {
    'DDI': torch.arange(g.number_of_edges('DDI')),
    'DPI': torch.arange(g.number_of_edges('DPI')),
    'PPI': torch.arange(g.number_of_edges('PPI'))
}
valid_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
valid_dataloader = dgl.dataloading.EdgeDataLoader(
    g, valid_eid, valid_sampler,
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=4)

for input_nodes, positive_graph, negative_graph, blocks in tqdm.tqdm(valid_dataloader):
    pos_score, neg_score = model(g, negative_graph, blocks, input_features)
