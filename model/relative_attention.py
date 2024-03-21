from torch import randn, tril, ones
from torch.nn import Module, Linear, Dropout, Parameter
from torch.nn.functional import pad, softmax

from math import sqrt


class RelativeAttention(Module):
    def __init__(self, d_model:int, max_seq_len:int):
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

        self.out = Linear(d_model, d_model)

        mask = tril(ones(max_seq_len, max_seq_len)) \
            .unsqueeze(0) \
            .unsqueeze(0)

        self.register_buffer('mask', mask)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        mask = self.mask[:,:, :seq_len, :seq_len]

        q = self.Q(x) \
            .reshape(batch_size, seq_len, self.num_heads, -1) \
            .transpose(1, 2)

        v = self.V(x) \
            .reshape(batch_size, seq_len, self.num_heads, -1) \
            .transpose(1, 2)

        k_t = self.K(x) \
            .reshape(batch_size, seq_len, self.num_heads, -1) \
            .permute(0, 2, 3, 1)

        s_rel = self.s_rel(q, seq_len)

        qk_t = q @ k_t
        logits = (qk_t + s_rel) / self.h_factor

        masked = logits.masked_fill(mask==0, -1e9)

        attn = softmax(masked, dim=-1) @ v

        out = attn \
            .transpose(1, 2) \
            .reshape(batch_size, seq_len, -1)
        
        x = self.out(out)

        return x

    
    def s_rel(self, q, seq_len):
        # Pad
        start = max(0, self.max_seq_len - seq_len)
        left_embedding = self.Er[start:,:] \
            .transpose(0, 1)

        qE = q @ left_embedding
        padded = pad(qE, (1, 0))

        # Skew
        dim1, dim2, dim3, dim4 = padded.shape
        padded = padded.reshape(dim1, dim2, dim4, dim3)

        s_rel = padded[:,:,1:,:]

        return s_rel