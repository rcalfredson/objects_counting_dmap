import torch
import random

x = torch.rand((10, 3, 4, 4))

with open('testOrigTensor', 'w') as myF:
    print('original x:', x, file=myF)
indexOrder = list(range(x.shape[0]))
random.shuffle(indexOrder)
indexOrder = torch.LongTensor(indexOrder)
print('reorder index:', indexOrder)
y = torch.zeros_like(x)
y = x[indexOrder]
print('after shuffle:', y)
print('from y:', y[0])
print('from x:', x[indexOrder[0]])
print(indexOrder)
