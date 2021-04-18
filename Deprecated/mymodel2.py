import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import pickle

# rel_names = [('drug', 'DDI', 'drug'), ('drug', 'DPI', 'protein'), ('protein', 'PPI', 'protein')]
rel_names = ['DDI', 'DPI', 'PPI']


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


# class StochasticTwoLayerRGCN(nn.Module):
#     def __init__(self, in_feat=1024, out_feat=1024):
#         super().__init__()
#         self.layer = dglnn.HeteroGraphConv({
#             # rel: dglnn.GraphConv(in_feat, out_feat, norm='right')
#             rel: dglnn.conv.SAGEConv(in_feat, out_feat, 'gcn', activation=F.relu)
#             for rel in rel_names
#         }, aggregate='sum')
#         self.pred = ScorePredictor()
#
#     def forward(self, g, neg_g, blocks):
#         h = self.layer(g, x)
#         return self.pred(g, h), self.pred(neg_g, h)


def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - neg_score.view(n_edges, -1) + pos_score.unsqueeze(1)).clamp(min=0).mean()


with open('./assets/g.pkl', 'rb') as f:
    g = pickle.load(f)

node_features = {
    'drug': torch.randn((g.number_of_nodes('drug'), 1024)),
    'protein': torch.randn((g.number_of_nodes('protein'), 1024))
}

train_eid = {
    'DDI': torch.arange(g.number_of_edges('DDI')),
    'DPI': torch.arange(g.number_of_edges('DPI')),
    'PPI': torch.arange(g.number_of_edges('PPI'))
}


class NegativeSampler(object):
    def __init__(self, g, k):
        # caches the probability distribution
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for etype in g.canonical_etypes}
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights.multinomial(len(src), replacement=True)
            result_dict[etype] = (src, dst)
        return result_dict
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)
dataloader = dgl.dataloading.EdgeDataLoader(
    g, train_eid, sampler,
    negative_sampler=NegativeSampler(g, 5),
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0)


def main():
    model = StochasticTwoLayerRGCN()
    # model = model.cuda()
    opt = torch.optim.Adam(model.parameters())

    # for epoch in range(10):
    #     for input_nodes, positive_graph, negative_graph, blocks in dataloader:
    #         blocks = [b.to(torch.device('cuda')) for b in blocks]
    #         positive_graph = positive_graph.to(torch.device('cuda'))
    #         negative_graph = negative_graph.to(torch.device('cuda'))
    #
    #         input_features = blocks[0].srcdata['features']
    #         edge_labels = edge_subgraph.edata['labels']
    #         edge_predictions = model(edge_subgraph, blocks, input_features)
    #         loss = compute_loss(edge_labels, edge_predictions)
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    for epoch in range(10):
        for input_nodes, positive_graph, negative_graph, blocks in dataloader:
            pos_score, neg_score = model(positive_graph, negative_graph, node_features, blocks)
            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item())


if __name__ == '__main__':
    main()
    print("Done!")
