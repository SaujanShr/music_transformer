from torch.nn import Module, Embedding

class WordEmbedding(Module):
    def __init__(self, vocab_size:int, d_model:int):
        '''
        Initialise the word embedding.

        Parameters:
            vocab_size (int): The number of tokens in the vocabulary.
            d_model (int): The embedding size.
        '''
        super().__init__()
        
        self.embed = Embedding(vocab_size, d_model)

    def forward(self, x):
        '''
        Forward pass of the word embedding.

        Parameters:
            x (tensor[batch_size, seq_len]): The input sequence.

        Returns:
            x (tensor[batch_size, seq_len, d_model]): The output data.
        '''
        x = self.embed(x)
        # x.shape = (batch_size, seq_len, d_model)
        
        return x
