from torch.nn import Module, Embedding

class WordEmbedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int = 512):
        super().__init__()

        self.embed = Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embed(x)
