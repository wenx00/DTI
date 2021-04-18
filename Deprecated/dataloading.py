import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import pickle

with open('g.pkl', 'rb') as f:
    g = pickle.load(f)


class RelGraphConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(1024, 1024, norm='right', weight=True, bias=True)
            for rel in ['DDI', 'DPI', 'PPI']
        })


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(1024, 1024, 'sum'))
        self.layers.append(dglnn.SAGEConv(1024, 1024, 'sum'))

    def forward(self, blocks, x):
        for index, (layer, block) in enumerate(zip(self.layers, blocks)):
            x = layer(block, x)
        return x


class ScorePredictor(nn.Module):
    def forward(self, subgraph, x):
        with subgraph.local_scope():
            subgraph.ndata['x'] = x
            subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))
            return subgraph.edata['score']


node_features = {
    'drug': torch.randn((g.number_of_nodes('drug'), 1024)),
    'protein': torch.randn((g.number_of_nodes('protein'), 1024))
}

train_eid = {
    'DDI': torch.arange(g.number_of_edges('DDI')),
    'DPI': torch.arange(g.number_of_edges('DPI')),
    'PPI': torch.arange(g.number_of_edges('PPI'))
}
sampler = dgl.dataloading.MultiLayerNeighborSampler([None])
neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)
train_dataloader = dgl.dataloading.EdgeDataLoader(
    g, train_eid, sampler,
    negative_sampler=neg_sampler,
    batch_size=1024, shuffle=True, drop_last=False, num_workers=4)

net = Net()
predictor = ScorePredictor()
optimizer = torch.optim.Adam(list(net.parameters()) + list(predictor.parameters()))

for epoch in range(20):
    net.train()
    for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(train_dataloader):
        inputs = node_features[input_nodes].cuda()
        outputs = net(blocks, inputs)
        pos_score = predictor(pos_graph, outputs)
        neg_score = predictor(neg_graph, outputs)

        score = torch.cat([pos_score, neg_score])
        label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(score, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)
