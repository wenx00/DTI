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
from sklearn.metrics import roc_auc_score
import tqdm

# tmp
import sys

with open('./assets/g_feat.pkl', 'rb') as f:
    g = pickle.load(f)

train_eid = {
    'DDI': torch.arange(g.number_of_edges('DDI')),
    'DPI': torch.arange(g.number_of_edges('DPI')),
    'PPI': torch.arange(g.number_of_edges('PPI'))
}
train_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
train_dataloader = dgl.dataloading.EdgeDataLoader(
    g, train_eid, train_sampler,
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(10),
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=2)
valid_eid = {
    'PPI': torch.arange(g.number_of_edges('PPI')),
    'DDI': torch.arange(g.number_of_edges('DDI')),
    'DPI': torch.arange(g.number_of_edges('DPI'))
}
valid_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
valid_dataloader = dgl.dataloading.EdgeDataLoader(
    g, valid_eid, valid_sampler,
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=2)

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
            rel: dglnn.GraphConv(in_features, out_features, norm='right', activation=F.relu)
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
    # positive logits
    p = torch.cat(
        (pos_score[('drug', 'DPI', 'protein')],
         pos_score[('drug', 'DDI', 'drug')],
         pos_score[('protein', 'PPI', 'protein')]
         ), dim=0).cuda()
    # negative logits
    n = torch.cat(
        (neg_score[('drug', 'DPI', 'protein')],
         neg_score[('drug', 'DDI', 'drug')],
         neg_score[('protein', 'PPI', 'protein')]
         ), dim=0).cuda()
    # loss function
    l = nn.CrossEntropyLoss().cuda()
    # positive_score
    a = l(torch.unsqueeze(p.squeeze(), 0), torch.LongTensor([1]).cuda())
    # negative_score
    b = l(torch.unsqueeze(n.squeeze(), 0), torch.LongTensor([0]).cuda())
    return torch.mean(torch.cat((a.view(1), b.view(1)), dim=0)), a, b


for epoch in range(20):
    # train model
    model.train()
    print('===' * 10)
    print('Epoch {} train started.'.format(epoch))
    t0 = time.time()
    for input_nodes, positive_graph, negative_graph, blocks in tqdm.tqdm(train_dataloader):
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        positive_graph = positive_graph.to(torch.device('cuda'))
        negative_graph = negative_graph.to(torch.device('cuda'))
        input_features = blocks[0].srcdata['feat']
        pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
        loss, a, b = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("Loss: {}, p_loss: {}, n_loss: {}".format(loss.item(), a.item(), b.item()))
    print('Epoch {} train finished, and takes {}s.'.format(epoch, time.time() - t0))

    # val model
    model.eval()
    print('--' * 10)
    print('Epoch {} valid started.'.format(epoch))
    t1 = time.time()
    for input_nodes, positive_graph, negative_graph, blocks in tqdm.tqdm(valid_dataloader):
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        positive_graph = positive_graph.to(torch.device('cuda'))
        negative_graph = negative_graph.to(torch.device('cuda'))
        input_features = blocks[0].srcdata['feat']
        pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)

        # positive logits
        p = torch.cat(
            (pos_score[('drug', 'DPI', 'protein')],
             pos_score[('drug', 'DDI', 'drug')],
             pos_score[('protein', 'PPI', 'protein')]
             ), dim=0).cuda()
        # negative logits
        n = torch.cat(
            (neg_score[('drug', 'DPI', 'protein')],
             neg_score[('drug', 'DDI', 'drug')],
             neg_score[('protein', 'PPI', 'protein')]
             ), dim=0).cuda()

        p = p.squeeze().cpu().detach().numpy()
        n = n.squeeze().cpu().detach().numpy()
        y_scores = np.concatenate((p, n), axis=0)
        y_true = np.concatenate((np.ones_like(p), np.zeros_like(n)), axis=0)
        print("AUCROC: {}".format(roc_auc_score(y_true, y_scores)))

    print('Epoch {} valid finished, and takes {}s.'.format(epoch, time.time() - t1))

torch.save(model.state_dict(), './assets/model_lnk.pt')
