from torch import tensor
from time import time

class Generator:
    def __init__(self, samples, max_seq_len, device):
        '''
        Initialise the sample generator.

        Parameters:
            samples (list[list[int]]): The training samples.
            max_seq_len (int): The maximum input sequence length.
            device (DeviceLikeType): The device the model runs on (CPU/GPU).
        '''
        self.samples = [
            sample for sample in samples 
            if len(sample) > max_seq_len
        ]
        self.max_seq_len = max_seq_len
        self.device = device

    def _tokens_to_learning_data(self, sample):
        '''
        Get inputs and labels from a training sample.

        Parameters:
            sample(list[int]): The training sample.

        Returns:
            inputs (tensor[num_inputs, max_seq_len]): The inputs.
            labels (tensor[num_labels, max_seq_len]): The labels.
        '''
        inputs = []
        labels = []

        for i in range(len(sample) - (self.max_seq_len+1)):
            inputs.append(sample[i:i+self.max_seq_len])
            labels.append(sample[i+1:i+self.max_seq_len+1])

        inputs = tensor(inputs).to(self.device)
        labels = tensor(labels).to(self.device)

        return inputs, labels

    def generate(self):
        '''
        Generate an input and label pair from the training samples.

        Returns:
            input (tensor[max_seq_len]): The input.
            label (tensor[max_seq_len]): The label.
        '''
        for i in range(len(self.samples)):
            inputs, labels = self._tokens_to_learning_data(self.samples[i])

            start = time()

            for j in range(len(inputs)):
                yield inputs[j], labels[j]

            end = time()

            print(f"\tsample:{i} sample_size:{len(inputs)} time_taken:{round(end - start, 2)}s")