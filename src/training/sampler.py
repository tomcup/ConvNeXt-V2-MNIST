import torch
import numpy as np

class ResumableRandomSubsetSampler(torch.utils.data.Sampler):
    """每个 epoch 可重新随机抽样的 sampler，支持 persistent workers"""
    def __init__(self, total_size, num_samples):
        self.total_size = total_size
        self.num_samples = num_samples
        self.indices = self._random_sample()

    def _random_sample(self):
        return np.random.choice(self.total_size, self.num_samples, replace=False)

    def resample(self):
        self.indices = self._random_sample()

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return self.num_samples