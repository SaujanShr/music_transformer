from torch import tensor

class Generator:
    def __init__(self, samples, max_seq_len):
        self.samples = samples
        self.max_seq_len = max_seq_len

    def _tokens_to_learning_data(self, sample):
        n = len(sample)

        if n <= self.max_seq_len:
            return ([], [])

        inputs = []
        labels = []

        for i in range(n - (self.max_seq_len+1)):
            inputs.append(sample[i:i+self.max_seq_len])
            labels.append(sample[i+1:i+self.max_seq_len+1])

        return tensor(inputs), tensor(labels)

    def generate(self):
        for i in range(len(self.samples)):
            inputs, labels = self._tokens_to_learning_data(self.samples[i])
            
            for j in range(len(inputs)):
                yield inputs[j], labels[j]

            print(f"sample:{i} sample_size:{j}")