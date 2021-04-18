import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import GraphConv
import numpy as np
import tqdm


def build_karate_club_graph():
    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints.
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
                    10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
                    25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
                    32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
                    33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
                    5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
                    24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
                    29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
                    31, 32])
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    return dgl.DGLGraph((u, v))


g = build_karate_club_graph()


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.num_classes = num_classes
        self.layer1 = GraphConv(in_feats, hidden_feats)
        self.layer2 = GraphConv(hidden_feats, num_classes)

    def forward(self, inputs):
        h = self.layer1(g, inputs)
        h = nn.ReLU(h)
        h = self.layer2(g, h)
        return h


train_nid = torch.arange(g.number_of_nodes())

node_features = nn.Embedding(g.number_of_nodes(), 5)
net = GCN(5, 5, 2)
sampler = dgl.dataloading.MultiLayerNeighborSampler([None, None])
train_dataloader = dgl.dataloading.NodeDataLoader(
    g, train_nid, sampler,
    batch_size=2,
    shuffle=True,
    drop_last=False,
    num_workers=2
)

optimizer = torch.optim.Adam(net.parameters())

# for epoch in range(10):
#     net.train()
#
#     with tqdm.tqdm_notebook(train_dataloader) as tq:
#         for step, (input_nodes, output_nodes, blocks) in enumerate(tq):
#             inputs=node_features[input_nodes]


# building the layer
layer = dglnn.HeteroGraphConv({
    'buys': dglnn.conv.SAGEConv(3, 100, 'gcn',
                                feat_drop=0.5, activation=F.relu),
    'bought-by': dglnn.conv.SAGEConv(5, 100, 'gcn',
                                     feat_drop=0.5, activation=F.relu)},
    aggregate='sum')

# assigning features
item_feat = torch.tensor(item_feat)
user_feat = torch.ones((g.number_of_nodes('user')), 5)
features = {'item': item_feat, 'user': user_feat}

h = features
out = layer(g, h)