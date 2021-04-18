import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import time
import dgl.nn.pytorch as dglnn
import pickle
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score
import tqdm

cuda = torch.cuda.is_available()


class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = torch.empty(3, 2048)
        nn.init.xavier_uniform_(self.params, gain=nn.init.calculate_gain('relu'))
        # self.rels = nn.ParameterDict({
        #     'DPI': nn.Parameter(torch.tensor(self.params[0], requires_grad=True)),
        #     'DDI': nn.Parameter(torch.tensor(self.params[1], requires_grad=True)),
        #     'PPI': nn.Parameter(torch.tensor(self.params[2], requires_grad=True))
        # })
        self.rels = nn.ParameterDict({
            'DPI': nn.Parameter(self.params[0].clone().detach().requires_grad_(True)),
            'DDI': nn.Parameter(self.params[1].clone().detach().requires_grad_(True)),
            'PPI': nn.Parameter(self.params[2].clone().detach().requires_grad_(True)),
        })
        # self.rels = nn.ParameterDict({
        #     'DPI': nn.Parameter(torch.ones_like(self.params[0], requires_grad=False)),
        #     'DDI': nn.Parameter(torch.ones_like(self.params[1], requires_grad=False)),
        #     'PPI': nn.Parameter(torch.ones_like(self.params[2], requires_grad=False))
        # })

    def edge_func_ddi(self, edges):
        head = edges.src['x']
        tail = edges.dst['x']
        score = head * self.rels['DDI'] * tail
        score = torch.sum(score, dim=-1)
        score = torch.clamp(score, min=0, max=1)
        return {'score': score}

    # def edge_func_dpi(self, edges):
    #     # todo: what is this # distmult
    #     head = edges.src['x']
    #     tail = edges.dst['x']
    #     score = head * self.rels['DPI'] * tail
    #     return {'score': torch.sum(score, dim=-1)}

    def edge_func_dpi(self, edges):
        head = edges.src['x']
        tail = edges.dst['x']
        score = head * self.rels['DPI'] * tail
        score = torch.sum(score, dim=-1)
        score = torch.clamp(score, min=0, max=1)
        return {'score': score}

    def edge_func_ppi(self, edges):
        head = edges.src['x']
        tail = edges.dst['x']
        score = head * self.rels['PPI'] * tail
        score = torch.sum(score, dim=-1)
        score = torch.clamp(score, min=0, max=1)
        return {'score': score}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(self.edge_func_ddi, etype=('drug', 'DDI', 'drug'))
            edge_subgraph.apply_edges(self.edge_func_dpi, etype=('drug', 'DPI', 'protein'))
            edge_subgraph.apply_edges(self.edge_func_dpi, etype=('protein', 'PPI', 'protein'))
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


def compute_loss_old(pos_score, neg_score):  # all 3 rel
    # positive_logits # todo: what is this
    positive_logits = torch.cat(
        (pos_score[('drug', 'DPI', 'protein')],
         pos_score[('drug', 'DDI', 'drug')],
         pos_score[('protein', 'PPI', 'protein')]
         ), dim=0)
    # negative_logits
    negative_logits = torch.cat(
        (neg_score[('drug', 'DPI', 'protein')],
         neg_score[('drug', 'DDI', 'drug')],
         neg_score[('protein', 'PPI', 'protein')]
         ), dim=0)
    # loss function
    loss_func = nn.CrossEntropyLoss()
    loss_func.cuda()
    # positive_score
    positive_score = loss_func(torch.unsqueeze(positive_logits, 0), torch.cuda.LongTensor([1]))
    # negative_score
    negative_score = loss_func(torch.unsqueeze(negative_logits, 0), torch.cuda.LongTensor([0]))
    return torch.mean(torch.cat((positive_score.view(1), negative_score.view(1)), dim=0)), \
           positive_score, negative_score


def compute_loss_good(pos_score, neg_score):
    # positive_logits # todo: what is this
    positive_logits = pos_score[('drug', 'DPI', 'protein')].cuda()
    # negative_logits
    negative_logits = neg_score[('drug', 'DPI', 'protein')].cuda()
    # loss function
    loss_func = nn.MSELoss()
    loss_func.cuda()
    # positive_score
    positive_score = loss_func(positive_logits, torch.ones_like(positive_logits))
    # negative_score
    negative_score = loss_func(negative_logits, torch.zeros_like(negative_logits))
    return torch.mean(torch.cat((positive_score.view(1), negative_score.view(1)), dim=0)), \
           positive_score, negative_score


def compute_loss(pos_score, neg_score):
    # positive_logits # todo: what is this
    positive_logits = torch.cat(
        (pos_score[('drug', 'DPI', 'protein')],
         pos_score[('drug', 'DDI', 'drug')],
         pos_score[('protein', 'PPI', 'protein')]
         ), dim=0).cuda()
    # negative_logits
    negative_logits = torch.cat(
        (neg_score[('drug', 'DPI', 'protein')],
         neg_score[('drug', 'DDI', 'drug')],
         neg_score[('protein', 'PPI', 'protein')]
         ), dim=0).cuda()
    # loss function
    loss_func = nn.MSELoss()  # torch.nn.BCEWithLogitsLoss
    loss_func.cuda()
    # positive_score
    positive_score = loss_func(positive_logits, torch.ones_like(positive_logits))
    # negative_score
    negative_score = loss_func(negative_logits, torch.zeros_like(negative_logits))
    return torch.mean(torch.cat((positive_score.view(1), negative_score.view(1)), dim=0)), \
           positive_score, negative_score


feature_size = 2048
# dropout_rate = args['dropout_rate']
batch_size = 4086  # 4086 12000
learning_rate = 0.0001
### --------- preparing data--------- ###
with open('./assets/g_feature.pkl', 'rb') as f:
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
# g.ndata['feat'] = node_features
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
# with open('./assets/valid_eid.pkl', 'rb') as v:
#     valid_eid = pickle.load(v)

rel_names = ['DDI', 'DPI', 'PPI']
checkpoint = torch.load('./assets/model_all_loss_0.966.pt')
# valid_eid = checkpoint['valid_eid']
# train_eid = checkpoint['train_eid']
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

# model.load_state_dict(torch.load('./assets/model_all_loss_0.96.pt'))

model = model.cuda()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)


# model.load_state_dict(checkpoint['model_state_dict'])
# opt.load_state_dict(checkpoint['optimizer_state_dict'])

def show_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)


### --------- starting epochs --------- ###
best_acu = 0.75
current_acu = 0
for epoch in range(1, 400):
    # train model
    model.train()
    print('====' * 10)
    print('Epoch {} train started.'.format(epoch))
    t0 = time.time()
    with tqdm.tqdm(train_dataloader) as tq_train:  # tq_train is tqdm for train
        for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(tq_train):
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            positive_graph = positive_graph.to(torch.device('cuda'))
            negative_graph = negative_graph.to(torch.device('cuda'))
            input_features = blocks[0].srcdata['feat']
            pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
            loss, p_loss, n_loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tq_train.set_postfix_str(
                'Loss: {:.5f}, p_loss: {:.5f}, n_loss: {:.5f}'.format(loss.item(), p_loss.item(), n_loss.item()),
                refresh=False)
    print('Epoch {} train finished, and takes {:.2f}s.'.format(epoch, time.time() - t0))

    # val model
    model.eval()
    print('--' * 10)
    print('Epoch {} valid started.'.format(epoch))
    t1 = time.time()
    with tqdm.tqdm(valid_dataloader) as tq_valid:  # tq_valid is tqdm for valid
        for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(tq_valid):
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            positive_graph = positive_graph.to(torch.device('cuda'))
            negative_graph = negative_graph.to(torch.device('cuda'))
            input_features = blocks[0].srcdata['feat']
            pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
            p_score = pos_score[('drug', 'DPI', 'protein')]
            n_score = neg_score[('drug', 'DPI', 'protein')]
            # positive logits
            p_score = p_score.squeeze().cpu().detach().numpy()
            # negative logits
            n_score = n_score.squeeze().cpu().detach().numpy()
            y_scores = np.concatenate((p_score, n_score), axis=0)
            y_trues = np.concatenate((np.ones_like(p_score), np.zeros_like(n_score)), axis=0)
            current_acu = roc_auc_score(y_trues, y_scores)
            # import pdb; pdb.set_trace()
            # print("ROCAUC: {}".format(current_acu))
            tq_valid.set_postfix_str("ROCAUC: {:.5f}".format(current_acu), refresh=False)

    print('Epoch {} valid finished, and takes {:.2f}s.'.format(epoch, time.time() - t1))
    ### --------- saving model --------- ###
    if best_acu == 1.1:
        best_acu = current_acu
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
            'g_feature': g,
            'train_eid': train_eid,
            'valid_eid': valid_eid
        }, './assets/model_all_loss_{}.pt'.format(current_acu))
