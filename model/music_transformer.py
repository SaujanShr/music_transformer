from torch.nn import Module, Softmax
from torch import long, cat
from torch.distributions import OneHotCategorical

from common.common import DEVICE
from model.decoder import Decoder

class MusicTransformer(Module):
    def __init__(self, vocab_size, 
        d_model=512, d_ff=2048,
        num_layers=8, dropout=0.1,
        max_seq_len=100):
        super().__init__()

        if d_model % 64:
            raise ValueError("Invalid embedding dimensionality")

        self.vocab_size = vocab_size

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
    
    def set_end_token(self, token):
        self.end_token = token

    def forward(self, x):
        x = self.word_embedding(x)
        x = self.positional_embedding(x)
        x = self.dropout(x)

        x = self.decoder(x)

        x = self.out(x)

        return x

    def generate(self, primer=[], max_seq_len=1024):
        arr = primer
        res = primer

        for i in range(max_seq_len):
            result = self.forward(arr).softmax(-1)

            pdf = OneHotCategorical(prob=result[:, -1])

            result = pdf.sample().argmax(-1).unsqueeze(-1)

            decode_array = cat((decode_array, result), dim=-1)
            result_array = cat((result_array, result), dim=-1)

        result_array = result_array[0]

        return result_array