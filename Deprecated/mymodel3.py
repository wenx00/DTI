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

with open('./assets/g.pkl', 'rb') as f:
    g = pickle.load(f)

drug_features = torch.empty((g.number_of_nodes('drug'), 1024))
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
train_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
train_dataloader = dgl.dataloading.EdgeDataLoader(
    g, train_eid, train_sampler,
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=4)

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


# model = StochasticTwoLayerRGCN(1024, 1024, 1024)
model = Model(1024, 1024, rel_names)
model = model.cuda()
opt = torch.optim.Adam(model.parameters())


def compute_loss(pos_score, neg_score):
    p = torch.cat(
        (pos_score[('drug', 'DPI', 'protein')],
         pos_score[('drug', 'DDI', 'drug')],
         pos_score[('protein', 'PPI', 'protein')]
         ), dim=0).cuda()
    n = torch.cat(
        (neg_score[('drug', 'DPI', 'protein')],
         neg_score[('drug', 'DDI', 'drug')],
         neg_score[('protein', 'PPI', 'protein')]
         ), dim=0).cuda()
    l = nn.CrossEntropyLoss().cuda()
    a = l(torch.unsqueeze(p.squeeze(), 0), torch.LongTensor([1]).cuda())
    b = l(torch.unsqueeze(n.squeeze(), 0), torch.LongTensor([0]).cuda())
    return torch.mean(torch.cat((a.view(1), b.view(1)), dim=0))


for epoch in range(20):
    model.train()
    print('--' * 10)
    print('Epoch {} started.'.format(epoch))
    t0 = time.time()
    for input_nodes, positive_graph, negative_graph, blocks in tqdm.tqdm(train_dataloader):
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        positive_graph = positive_graph.to(torch.device('cuda'))
        negative_graph = negative_graph.to(torch.device('cuda'))
        input_features = blocks[0].srcdata['feat']
        pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
        loss,= compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("Loss: {}.".format(loss.item()))

    print('Epoch {} finished, and takes {}s.'.format(epoch, time.time() - t0))



torch.save(model.state_dict(), './assets/model.pt')
torch.save(model.state_dict(), 'model.pt')
