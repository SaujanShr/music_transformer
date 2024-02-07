from torch.nn import Module, Embedding

class WordEmbedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int = 512):
        super(WordEmbedding, self).__init__()

        self.embed = Embedding(num_embeddings, embedding_dim)

    def forward(self, input_vector):
        return self.embed(input_vector)