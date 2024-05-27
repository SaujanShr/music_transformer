from common import config

from torch import no_grad, tensor, cat
from torch.distributions import Categorical

class Generator:
    def __init__(self, model, 
            device=config.DEVICE, 
            max_seq_len=config.MAX_SEQUENCE_LENGTH,
            start_mapping=0, end_mapping=1):
        '''
        Initialise the generator.

        Parameters:
            model (MusicTransformer): The transformer model.
            device (DeviceLikeType): The device the model runs on (CPU/GPU).
            max_seq_len (int): The maximum input sequence length.
            start_mapping (int): Mapping for the START token.
            end_mapping (int): Mapping for the END token.
        '''
        self.model = model
        model.eval()

        self.device = device
        self.max_seq_len = max_seq_len

        self.start = start_mapping
        self.end = end_mapping

    def generate(self, primer, max_gen_len=config.MAX_GENERATION_LENGTH):
        '''
        Generate an integer token sequence continuing from the primer.

        Parameters:
            primer (list[int]): The starting integer tokens of the generated sequence.
            max_gen_len (int): The maximum generated sequence length.

        Returns:
            result (list[int]): The generated integer token sequence.
        '''
        input_vector = tensor(primer[-self.max_seq_len:]) \
            .unsqueeze(0) \
            .to(self.device)
    
        result = primer

        print("Starting generating")

        with no_grad():
            for i in range(min(len(primer), self.max_seq_len), max_gen_len):
                print(f"Generating token:{i}")
                y = self.model(input_vector)
                probs = self.model.softmax(y[:, -1, :])

                prediction = Categorical(probs).sample()

                if prediction == self.end:
                    result.append(prediction.item())
                    print("Finished generating")
                    return result

                input_vector = cat([input_vector, prediction.view(1, 1)], dim=-1)[:, -self.max_seq_len:]
                result.append(prediction.item())

        print("Finished generating")
        return result