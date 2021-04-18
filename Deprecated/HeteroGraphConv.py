import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np

edges1 = [(0, 1)]
edges2 = [(0, 0)]
edges3 = [(0, 0)]

g = dgl.heterograph({
    ('user', 'follows', 'user'): edges1,
    ('user', 'plays', 'game'): edges2,
    ('store', 'sells', 'game'): edges3})

net = dglnn.HeteroGraphConv({
    'follows': dglnn.GraphConv(5, 5),
    'plays': dglnn.GraphConv(5, 5),
    'sells': dglnn.GraphConv(5, 5)},
    aggregate='sum')

user_feat = torch.randn((g.number_of_nodes('user'), 5))
store_feat = torch.randn((g.number_of_nodes('store'), 5))
game_feat = torch.randn((g.number_of_nodes('game'), 5))
features = {
    'user': user_feat,
    'store': store_feat,
    'game': game_feat
}
print(features)
print("--" * 10)
out = net(g, features)
print(out)
print("--" * 10)
