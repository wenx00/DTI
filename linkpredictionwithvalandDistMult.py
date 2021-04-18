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


class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rel = nn.Parameter(torch.ones([1024], requires_grad=True))
        if cuda:
            self.rel.cuda()

    def edge_func(self, edges):
        # todo: what is this # distmult
        head = edges.src['x']
        tail = edges.dst['x']
        score = head * self.rel * tail
        # todo: use dot product # no
        return {'score': torch.sum(score, dim=-1)}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(self.edge_func, etype=etype) # improve performance # lambda edges: self.edge_func(edges)
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
         ), dim=0)
    if cuda:
        positive_logits.cuda()
    # negative_logits
    negative_logits = torch.cat(
        (neg_score[('drug', 'DPI', 'protein')],
         neg_score[('drug', 'DDI', 'drug')],
         neg_score[('protein', 'PPI', 'protein')]
         ), dim=0)
    if cuda:
        negative_logits.cuda()
    # loss function
    loss_func = nn.CrossEntropyLoss()
    if cuda:
        loss_func.cuda()
    # positive_score
    positive_score = loss_func(torch.unsqueeze(positive_logits.squeeze(), 0), torch.LongTensor([1]))
    # negative_score
    negative_score = loss_func(torch.unsqueeze(negative_logits.squeeze(), 0), torch.LongTensor([0]))
    return torch.mean(torch.cat((positive_score.view(1), negative_score.view(1)), dim=0)), \
           positive_score, negative_score


def main():
    feature_size = 1024
    # dropout_rate = args['dropout_rate']
    batch_size = 1024
    learning_rate = 0.0003
    ### --------- preparing data--------- ###
    with open('./assets/g.pkl', 'rb') as f:
        g = pickle.load(f)
    # assign features
    drug_features = torch.empty((g.number_of_nodes('drug'), feature_size))
    protein_features = torch.empty((g.number_of_nodes('protein'), feature_size))
    nn.init.xavier_uniform_(drug_features, gain=nn.init.calculate_gain('relu'))
    nn.init.xavier_uniform_(protein_features, gain=nn.init.calculate_gain('relu'))
    node_features = {
        'drug': drug_features,
        'protein': protein_features
    }
    g.ndata['feat'] = node_features
    # split DPI eid
    eid_DPI = np.arange(g.number_of_edges('DPI'))
    np.random.shuffle(eid_DPI)
    train_eid_DPI = eid_DPI[len(eid_DPI) // 5:]
    valid_eid_DPI = eid_DPI[:len(eid_DPI) // 5]
    train_eid = {
        'DDI': torch.arange(g.number_of_edges('DDI')),
        'DPI': torch.LongTensor(train_eid_DPI),
        'PPI': torch.arange(g.number_of_edges('PPI'))
    }
    valid_eid = {
        # 'PPI': torch.arange(g.number_of_edges('PPI')),
        # 'DDI': torch.arange(g.number_of_edges('DDI')),
        'DPI': torch.LongTensor(valid_eid_DPI),
    }
    rel_names = ['DDI', 'DPI', 'PPI']
    ### --------- building model --------- ###
    train_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    train_neg_sampler = dgl.dataloading.negative_sampler.Uniform(10)
    train_dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid, train_sampler,
        negative_sampler=train_neg_sampler,
        batch_size=batch_size,  # 65536
        shuffle=True,
        drop_last=False,
        num_workers=4)
    valid_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    valid_neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    valid_dataloader = dgl.dataloading.EdgeDataLoader(
        g, valid_eid, valid_sampler,
        negative_sampler=valid_neg_sampler,
        batch_size=14582,
        shuffle=True,
        drop_last=False,
        num_workers=0)
    model = Model(feature_size, feature_size, rel_names)
    if cuda():
        model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ### --------- starting epochs --------- ###
    best_acu = 0.75
    current_acu = 0
    for epoch in range(400):
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
            p = pos_score[('drug', 'DPI', 'protein')]
            n = neg_score[('drug', 'DPI', 'protein')]
            # positive logits
            p = p.squeeze().cpu().detach().numpy()
            # negative logits
            n = n.squeeze().cpu().detach().numpy()
            y_scores = np.concatenate((p, n), axis=0)
            y_true = np.concatenate((np.ones_like(p), np.zeros_like(n)), axis=0)
            current_acu = roc_auc_score(y_true, y_scores)
            print("ROCAUC: {}".format(current_acu))
        print('Epoch {} valid finished, and takes {}s.'.format(epoch, time.time() - t1))
        ### --------- saving model --------- ###
        if current_acu > best_acu:
            best_acu = current_acu
            torch.save(model.state_dict(), './assets/model_{}.pt'.format(current_acu))


if __name__ == '__main__':
    global cuda
    cuda = torch.cuda.is_available()
    main()
