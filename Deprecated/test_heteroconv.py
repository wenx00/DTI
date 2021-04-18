import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import pickle
import mymodel2


with open('./assets/g.pkl', 'rb') as f:
    g = pickle.load(f)
rel_names = ['DDI', 'DPI', 'PPI']
node_features = {
    'drug': torch.randn((g.number_of_nodes('drug'), 10)),
    'protein': torch.randn((g.number_of_nodes('protein'), 10))
}
pred = mymodel2.ScorePredictor()
layer = dglnn.HeteroGraphConv({
    rel: dglnn.GraphConv(10, 10, norm='right')
    # rel: dglnn.conv.SAGEConv(in_feat, out_feat, 'gcn', activation=F.relu)
    for rel in rel_names
}, aggregate='sum')

out = layer(g,node_features)
out=pred(g,out)
print(out)