from math import ceil

from training.dataset import Dataset
from torch.utils.data import DataLoader

def get_data_loaders(samples, max_seq_len, batch_size, split, device):
    '''
    Get the training and validation loaders for the samples.

    Parameters:
        samples (list[list[int]]): The training samples.
        max_seq_len (int): The maximum input sequence length.
        batch_size (int): The size of the training batches.
        split (float): The fraction of training samples to validation samples.
        device (DeviceLikeType): The device the model runs on (CPU/GPU).

    Returns:
        training_loader (DataLoader): The dataloader for the training dataset.
        validation_loader (DataLoader): The dataloader for the validation dataset.
    '''
    cutoff = ceil(len(samples) * split)

    training_samples = samples[:cutoff]
    validation_samples = samples[cutoff:]

    training_dataset = Dataset(training_samples, max_seq_len, device)
    validation_dataset = Dataset(validation_samples, max_seq_len, device)

    training_loader = DataLoader(training_dataset, batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size)

    return training_loader, validation_loader