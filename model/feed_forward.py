from torch.nn import Module, Linear, ReLU

class FeedForward(Module):
    def __init__(self, d_model, d_ff):
        '''
        Initialise the feedforward layer.

        Parameters:
            d_model (int): The embedding size.
            d_ff (int): The feedforward output size.
        '''
        super().__init__()

        self.layer_1 = Linear(d_model, d_ff)
        self.layer_2 = Linear(d_ff, d_model)

        self.relu = ReLU()

    def forward(self, x):
        '''
        Forward pass of the feedforward layer.

        Parameters:
            x (tensor[batch_size, seq_len, d_model]): The input data.

        Returns:
            x (tensor[batch_size, seq_len, d_model]): The output data.
        '''
        x = self.layer_1(x)
        # x.shape = (batch_size, seq_len, d_ff)
        x = self.relu(x)
        
        x = self.layer_2(x)
        # x.shape = (batch_size, seq_len, d_model)

        return x
