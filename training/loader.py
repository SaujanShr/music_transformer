from math import ceil

from torch import Tensor

from training.dataset import Dataset
from torch.utils.data import DataLoader

def get_data_loaders(samples, max_seq_len, batch_size, split):
    cutoff = ceil(len(samples) * split)

    training_samples = samples[:cutoff]
    validation_samples = samples[cutoff:]

    training_dataset = Dataset(training_samples, max_seq_len)
    validation_dataset = Dataset(validation_samples, max_seq_len)

    training_loader = DataLoader(training_dataset, batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size)

    return training_loader, validation_loader