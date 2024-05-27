from torch import randn
from torch.nn import Module, Linear, Parameter
from torch.nn.functional import pad, softmax

from math import sqrt

class RelativeAttention(Module):
    def __init__(self, d_model:int, max_seq_len:int):
        '''
        Initialise the relative multi-head attention layer.

        Parameters:
            d_model (int): The embedding size.
            max_seq_len (int): The maximum input sequence length.
        '''
        super().__init__()

        d_head = 64

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.h_factor = sqrt(d_head)
        self.num_heads = d_model // d_head

        self.Q = Linear(d_model, d_model)
        self.V = Linear(d_model, d_model)
        self.K = Linear(d_model, d_model)

        self.Er = Parameter(randn(max_seq_len, d_head))
        # Er.shape = (max_seq_len, d_head)

        self.out = Linear(d_model, d_model)

    def forward(self, x, mask):
        '''
        Forward pass of the relative multi-head attention layer.

        Parameters:
            x (tensor[batch_size, seq_len, d_model]): The input data.
            mask (tensor[1, 1, seq_len, seq_len]): The self-attention mask.

        Returns:
            x (tensor[batch_size, seq_len, d_model]): The output data.
        '''
        batch_size, seq_len, _ = x.shape

        q = self.Q(x) \
            .reshape(batch_size, seq_len, self.num_heads, -1) \
            .transpose(1, 2)
        # q.shape = (batch_size, num_heads, seq_len, d_head)

        v = self.V(x) \
            .reshape(batch_size, seq_len, self.num_heads, -1) \
            .transpose(1, 2)
        # v.shape = (batch_size, num_heads, seq_len, d_head)

        k_t = self.K(x) \
            .reshape(batch_size, seq_len, self.num_heads, -1) \
            .permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)

        s_rel = self.s_rel(q, seq_len)
        # s_rel.shape = (batch_size, num_heads, seq_len, seq_len)

        qk_t = q @ k_t
        # qk_t.shape = (batch_size, num_heads, seq_len, seq_len)
        logits = (qk_t + s_rel) / self.h_factor

        masked = logits + mask.to(logits.dtype)
        attn = softmax(masked, -1) @ v

        x = attn \
            .transpose(1, 2) \
            .reshape(batch_size, -1, self.d_model)
        # x.shape = (batch_size, seq_len, d_model)
        
        x = self.out(x)

        return x

    
    def s_rel(self, q, seq_len):
        '''
        Get the relative positional embedding term.

        Parameters:
            q (tensor[batch_size, num_heads, seq_len, d_head]): The query vector.
            seq_len (int): The input sequence length.

        Returns:
            s_rel (tensor[batch_size, num_heads, seq_len, seq_len]): The relative positional embedding term.
        '''
        # Pad
        start = max(0, self.max_seq_len - seq_len)
        left_embedding = self.Er[start:,:] \
            .transpose(0, 1)
        # left_embedding.shape = (d_head, seq_len)

        qE = q @ left_embedding
        # qE.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = pad(qE, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, seq_len + 1)

        # Skew
        dim1, dim2, dim3, dim4 = padded.shape
        reshaped = padded.reshape(dim1, dim2, dim4, dim3)
        # reshaped.shape = (batch_size, num_heads, seq_len + 1, seq_len)

        s_rel = reshaped[:,:,1:,:]
        # s_rel.shape = (batch_size, num_heads, seq_len, seq_len)

        return s_rel