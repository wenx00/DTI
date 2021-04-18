import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import pickle
import itertools
import numpy as np
import tqdm

with open('./assets/g.pkl', 'rb') as f:
    g = pickle.load(f)

drug_features=torch.empty((g.number_of_nodes('drug'), 1024))
protein_features = torch.empty((g.number_of_nodes('protein'), 1024))
nn.init.xavier_uniform_(drug_features, gain=nn.init.calculate_gain('relu'))
nn.init.xavier_uniform_(protein_features, gain=nn.init.calculate_gain('relu'))

node_features = {
    'drug': drug_features,
    'protein': protein_features
}

g.ndata['feat'] = node_features

# print(g.ndata['feat'])
# ttt = input("OK")

train_eid = {
    'DDI': torch.arange(g.number_of_edges('DDI')),
    'DPI': torch.arange(g.number_of_edges('DPI')),
    'PPI': torch.arange(g.number_of_edges('PPI'))
}
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
dataloader = dgl.dataloading.EdgeDataLoader(
    g, train_eid, sampler,
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0)

rel_names = ['DDI', 'DPI', 'PPI']


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, hidden_feat, norm='right')
            for rel in rel_names
        })
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hidden_feat, out_feat, norm='right')
            for rel in rel_names
        })

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        return x


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


# model = StochasticTwoLayerRGCN(1024, 1024, 1024)
model = Model(1024, 1024, rel_names)
opt = torch.optim.Adam(model.parameters())


def compute_loss(pos_score, neg_score):
    # Margin loss
    # pos_score = np.asarray(list(itertools.chain.from_iterable(list(pos_score.values()))))
    # neg_score = np.asarray(list(itertools.chain.from_iterable(list(neg_score.values()))))
    # n_edges = pos_score.shape[0]
    # return (1 - neg_score.view(n_edges, -1) + pos_score.unsqueeze(1)).clamp(min=0).mean()
    p = torch.cat(
        (pos_score[('drug', 'DPI', 'protein')],
         pos_score[('drug', 'DDI', 'drug')],
         pos_score[('protein', 'PPI', 'protein')]
         ), dim=0)
    n = torch.cat(
        (neg_score[('drug', 'DPI', 'protein')],
         neg_score[('drug', 'DDI', 'drug')],
         neg_score[('protein', 'PPI', 'protein')]
         ), dim=0)
    # a = torch.cat((p, n), dim=0)
    # b = torch.cat((torch.ones_like(p), torch.zeros_like(n)), dim=0)
    l = nn.CrossEntropyLoss()
    a= l(torch.unsqueeze(p.squeeze(), 0), torch.LongTensor([1]))
    b= l(torch.unsqueeze(n.squeeze(), 0), torch.LongTensor([0]))
    return torch.mean(torch.cat((a.view(1), b.view(1)), dim=0))


for epoch in range(5):
    print('Epoch {}:'.format(epoch))
    for input_nodes, positive_graph, negative_graph, blocks in tqdm.tqdm(dataloader):
        input_features = blocks[0].srcdata['feat']
        pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
