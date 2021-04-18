import numpy as np
import itertools
import torch


d = {
    111: [1, 2],
    222: [3, 4]
}

print(d.values())
print(list(d.values()))
print(list(itertools.chain.from_iterable(list(d.values()))))

l= np.asarray(list(itertools.chain.from_iterable(list(d.values()))))


print(l)

print(np.asarray(itertools.chain.from_iterable(list(d.values()))))

input = torch.empty(5)
print(torch.ones_like(input))