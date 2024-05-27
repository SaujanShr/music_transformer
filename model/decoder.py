from model.decoder_layer import DecoderLayer

from torch.nn import Module, ModuleList

class Decoder(Module):
    def __init__(self, d_model, d_ff, num_layers, dropout, max_seq_len):
        '''
        Initialise the decoder.

        Parameters:
            d_model (int): The embedding size.
            d_ff (int): The feedforward output size.
            num_layers (int): The number of decoder layers.
            dropout (float): The dropout rate.
            max_seq_len (int): The maximum input sequence length.
        '''
        super().__init__()

        self.decoder_layers = ModuleList([
            DecoderLayer(d_model, d_ff, dropout, max_seq_len) 
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        '''
        Forward pass of the decoder.

        Parameters:
            x (tensor[batch_size, seq_len, d_model]): The input data.
            mask (tensor[1, 1, seq_len, seq_len]): The self-attention mask.

        Returns:
            x (tensor[batch_size, seq_len, d_model]): The output data.
        '''
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask)
        
        return x