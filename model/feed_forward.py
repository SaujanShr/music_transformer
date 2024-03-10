from torch.nn import Module, Linear, ReLU

class FeedForward(Module):
    def __init__(self, embedding_dim, ff_dim):
        super().__init__()

        self.layer_1 = Linear(embedding_dim, ff_dim)
        self.layer_2 = Linear(ff_dim, embedding_dim)

        self.relu = ReLU()

    def forward(self, x):
        return self.layer_2(self.relu(self.layer_1(x)))
