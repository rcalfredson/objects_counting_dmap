# source: https://discuss.pytorch.org/t/new-subset-every-epoch/85018
import torch
from torch.utils.data.sampler import Sampler

class RandomSampler(Sampler):
    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        permutations = []
        for _ in range(int(self.num_samples / n)):
            permutations += torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist()
        return iter(permutations)

    def __len__(self):
        return self.num_samples