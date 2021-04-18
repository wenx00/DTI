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

print(g)

print('---' * 20)
i = dgl.remove_edges(g, torch.arange(1950200), 'DDI')
i = dgl.remove_edges(i, torch.arange(217000), 'PPI')

print(i)

"""123"""
with open('./assets/i.pkl', 'wb') as fp:
    pickle.dump(i, fp)