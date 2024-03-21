from math import ceil

from torch import tensor
from torch.utils.data import IterableDataset

from training.generator import Generator

class Dataset(IterableDataset):
    def __init__(self, samples, max_seq_len):
        super().__init__()

        self.generator = Generator(samples, max_seq_len)

    def __iter__(self):
        return self.generator.generate()
