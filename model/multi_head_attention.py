from torch.nn import Module, Linear

class MultiheadAttention(Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        head_dim = embedding_dim // num_heads

        self.query = Linear(head_dim, head_dim, bias=False)
        self.key = Linear(head_dim, head_dim, bias=False)
        self.value = Linear(head_dim, head_dim, bias=False)

        self.out = Linear(num_heads*head_dim, embedding_dim)

    def forward(self, k, q, v, mask=None):
        pass

    def split_heads(self, x):
        batch_size, seq_length, embedding_dim = x.size()

        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    