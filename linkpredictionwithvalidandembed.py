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


with open('./assets/g.pkl', 'rb') as f:
    g = pickle.load(f)

eid_DPI = np.arange(g.number_of_edges('DPI'))
np.random.shuffle(eid_DPI)
train_eid_DPI = eid_DPI[len(eid_DPI) // 5:]
valid_eid_DPI = eid_DPI[:len(eid_DPI) // 5]

train_eid = {
    'DDI': torch.arange(g.number_of_edges('DDI')),
    'DPI': torch.LongTensor(train_eid_DPI),
    'PPI': torch.arange(g.number_of_edges('PPI'))
}
train_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
train_dataloader = dgl.dataloading.EdgeDataLoader(
    g, train_eid, train_sampler,
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(10),
    batch_size=65536,  # 65536 1024
    shuffle=True,
    drop_last=False,
    num_workers=4)
valid_eid = {
    # 'PPI': torch.arange(g.number_of_edges('PPI')),
    # 'DDI': torch.arange(g.number_of_edges('DDI')),
    'DPI': torch.LongTensor(valid_eid_DPI),
}
valid_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
valid_dataloader = dgl.dataloading.EdgeDataLoader(
    g, valid_eid, valid_sampler,
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
    batch_size=14582,
    shuffle=True,
    drop_last=False,
    num_workers=0)

rel_names = ['DDI', 'DPI', 'PPI']


def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""

    def __init__(self,
                 g,
                 embed_size,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            embed = nn.Parameter(th.Tensor(g.number_of_nodes(ntype), self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed

    def forward(self, block=None):
        """Forward computation
        Parameters
        ----------
        block : DGLHeteroGraph, optional
            If not specified, directly return the full graph with embeddings stored in
            :attr:`embed_name`. Otherwise, extract and store the embeddings to the block
            graph and return.
        Returns
        -------
        DGLHeteroGraph
            The block graph fed with embeddings.
        """
        return self.embeds


embed_layer = RelGraphEmbed(g, 1024)
embed_layer.cuda()
node_embed = embed_layer()


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
all_params = itertools.chain(model.parameters(), embed_layer.parameters())
opt = torch.optim.Adam(all_params, lr=0.001, weight_decay=5e-4)


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


for epoch in range(4000):
    # train model
    model.train()
    print('===' * 10)
    print('Epoch {} train started.'.format(epoch))
    t0 = time.time()

    for input_nodes, positive_graph, negative_graph, blocks in tqdm.tqdm(train_dataloader):
        emb = extract_embed(node_embed, input_nodes)
        if torch.cuda.is_available():
            emb = {k: e.cuda() for k, e in emb.items()}
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            positive_graph = positive_graph.to(torch.device('cuda'))
            negative_graph = negative_graph.to(torch.device('cuda'))

        pos_score, neg_score = model(positive_graph, negative_graph, blocks, emb)
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
        emb = extract_embed(node_embed, input_nodes)
        if torch.cuda.is_available():
            emb = {k: e.cuda() for k, e in emb.items()}
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            positive_graph = positive_graph.to(torch.device('cuda'))
            negative_graph = negative_graph.to(torch.device('cuda'))

        pos_score, neg_score = model(positive_graph, negative_graph, blocks, emb)
        p = pos_score[('drug', 'DPI', 'protein')]
        n = neg_score[('drug', 'DPI', 'protein')]
        # positive logits
        p = p.squeeze().cpu().detach().numpy()
        # negative logits
        n = n.squeeze().cpu().detach().numpy()
        y_scores = np.concatenate((p, n), axis=0)
        y_true = np.concatenate((np.ones_like(p), np.zeros_like(n)), axis=0)

        print("ROCAUC: {}".format(roc_auc_score(y_true, y_scores)))
    print('Epoch {} valid finished, and takes {}s.'.format(epoch, time.time() - t1))


# save model and embeddings
torch.save(model.state_dict(), './assets/model_embed.pt')
with open('./assets/node_embed.pkl', 'wb') as ef:
    pickle.dumps(node_embed, ef)
