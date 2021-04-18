import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import pickle


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class EntityClassify(nn.Module):
    def __init__(self,
                 g,
                 h_dim, out_dim,
                 num_bases,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(EntityClassify, self).__init__()
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        # self.embed_layer = RelGraphEmbed(g, self.h_dim)
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.h_dim, self.rel_names,
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=False))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names,
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    def forward(self, h, blocks):
        # if blocks is None:
        # full graph training
        # blocks = [self.g] * len(self.layers)
        # if h is None:
        # full graph training
        # h = self.embed_layer()
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h


class ScorePredictor(nn.Module):
    def forward(self, subgraph, x):
        with subgraph.local_scope():
            subgraph.ndata['x'] = x
            subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))
            return subgraph.edata['score']


def main():
    with open('./assets/g.pkl', 'rb') as f:
        g = pickle.load(f)

    node_features = {
        'drug': torch.randn((g.number_of_nodes('drug'), 1024)),
        'protein': torch.randn((g.number_of_nodes('protein'), 1024))
    }

    model = EntityClassify(g,
                           h_dim=1024,
                           out_dim=1024,
                           num_bases=-1,
                           num_hidden_layers=1,
                           dropout=0,
                           use_self_loop=False)

    sampler = dgl.dataloading.MultiLayerNeighborSampler([None])
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    train_eid = {
        'DDI': torch.arange(g.number_of_edges('DDI')),
        'DPI': torch.arange(g.number_of_edges('DPI')),
        'PPI': torch.arange(g.number_of_edges('PPI'))
    }
    train_dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid, sampler,
        negative_sampler=neg_sampler,
        batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # , weight_decay=args.l2norm)

    print("start training...")

    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(train_dataloader):
            logits = model(node_features[input_nodes], blocks)
            print(logits)
            pass


if __name__ == '__main__':
    main()
    print("OK")
