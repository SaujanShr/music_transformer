from torch.nn import Module, Softmax, Dropout, Linear
from torch import tril, ones

from common import config
from model.decoder import Decoder
from model.word_embedding import WordEmbedding
from model.positional_embedding import PositionalEmbedding

class MusicTransformer(Module):
    def __init__(self, vocab_size, 
        d_model=config.EMBEDDING_SIZE, d_ff=config.FEEDFORWARD_SIZE,
        num_layers=config.NUM_LAYERS, dropout=config.DROPOUT,
        max_seq_len=config.MAX_SEQUENCE_LENGTH,
        device=config.DEVICE):
        '''
        Initialise the Music Transformer.

        Parameters:
            vocab_size (int): The number of tokens in the vocabulary.
            d_model (int): The embedding size.
            d_ff (int): The feedforward output size.
            num_layers (int): The number of decoder layers.
            dropout (float): The dropout rate.
            max_seq_len (int): The maximum input sequence length.
            device (DeviceLikeType): The device the model runs on (CPU/GPU).
        '''
        super().__init__()

        if d_model % 64:
            raise ValueError("Invalid embedding dimensionality")
        
        self.device = device

        self.word_embedding = WordEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, max_seq_len)
        self.dropout = Dropout(dropout)

        self.decoder = Decoder( 
            d_model, d_ff, 
            num_layers, dropout, 
            max_seq_len
        )

        self.out = Linear(d_model, vocab_size)
        self.softmax = Softmax(dim = -1)

        self.register_buffer(
            "mask",
            tril(ones(max_seq_len, max_seq_len))
                .unsqueeze(0)
                .unsqueeze(0)
        )
        # mask.shape = (1, 1, seq_len, seq_len)

    def forward(self, x):
        '''
        Forward pass of the Music Transformer.

        Parameters:
            x (tensor[batch_size, seq_len]): The input sequence.

        Returns:
            x (tensor[batch_size, seq_len, vocab_size]): The output data.
        '''
        x = self.word_embedding(x)
        # x.shape = (batch_size, seq_len, d_model)
        x = self.positional_embedding(x)
        x = self.dropout(x)

        _, seq_len, _ = x.shape
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)

        x = self.decoder(x, mask)
        x = self.out(x)
        
        return x
