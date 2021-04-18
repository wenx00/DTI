import torch
import dgl

# import pickle

g = dgl.heterograph({
    ('user', 'follows', 'user'): ([0, 1], [1, 2]),
    ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
    ('developer', 'develops', 'game'): ([0, 1], [0, 1]),
})

print(g)
