from torch.nn import Module, Embedding

class WordEmbedding(Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        
        self.embed = Embedding(vocab_size, d_model)

    def forward(self, x):
        x = self.embed(x)

        return x
