import json
import os

import torch
import matplotlib.pyplot as plt

from data_loader import H5Dataset

dataset = H5Dataset('egg/train.h5')
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1)
eggCounts = []

if not os.path.exists('eggCountsRaw.json'):
    count = 0
    for image, label in dataloader:
        eggCounts.append((torch.sum(label) / 100).item())
        if count % 10 == 0:
            print('checking item %i' % count)
        count += 1
    with open('eggCountsRaw.json', 'w') as f:
        json.dump(eggCounts, f, ensure_ascii=False, indent=4)
else:
    with open('eggCountsRaw.json', 'r') as f:
        eggCounts = json.load(f)

print('eggCounts?', eggCounts[0:10])

plt.figure()
plt.hist(eggCounts, bins=30)
plt.title('Number of eggs per patch')
plt.xlabel('# eggs')
plt.xlabel('# patches')
plt.show()
