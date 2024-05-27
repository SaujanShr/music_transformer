from torch.utils.data import IterableDataset

from training.generator import Generator 

class Dataset(IterableDataset):
    def __init__(self, samples, max_seq_len, device):
        '''
        Initialise the generator dataset.

        Parameters:
            samples (list[list[int]]): The training samples.
            max_seq_len (int): The maximum input sequence length.
            device (DeviceLikeType): The device the model runs on (CPU/GPU).
        '''
        super().__init__()

        self.generator = Generator(samples, max_seq_len, device)

    def __iter__(self):
        '''
        Get the iterative sample generator function that yields an input and label tensor.

        Returns:
            generate (Generator[tuple[tensor[int],tensor[int]]]): The sample generator function.
        '''
        return self.generator.generate()
