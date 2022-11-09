import torch
import torch.nn as nn
from torch import Tensor
from sublayers.scaled_dot_product_attention import Attention

class MultiHead(nn.Module):
    """
    Applies self-attention to multiple heads
    """
    def __init__(self, num_heads, dim_model):
        super(MultiHead, self).__init__()
        
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_k = dim_model // num_heads

        assert (self.dim_k * num_heads == dim_model), "dim_model must be divisible by num_heads"

        self.attention = Attention()
        self.W_Q = nn.Linear(self.dim_model, self.num_heads * self.dim_k)
        self.W_K = nn.Linear(self.dim_model, self.num_heads * self.dim_k)
        self.W_V = nn.Linear(self.dim_model, self.num_heads * self.dim_k)
        self.W_concat = nn.Linear(dim_model, dim_model)

    def forward(self, Q: Tensor, K, V, mask=None):
        # Apply weight matrices
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        
        # Split tensors into num_heads heads
        # before: [batch_size, seq_len, dim_model]
        # after: [batch_size, num_heads, seq_len, dim_k]
        batch_size, seq_len, d_model = Q.size()
        Q = Q.view(batch_size, seq_len, self.num_heads, self.dim_k).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.dim_k).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.dim_k).transpose(1,2)

        # Apply scaled dot product attention
        out = self.attention(Q, K, V, mask=mask)

        # Concatenate
        # before: [batch_size, num_heads, seq_len, dim_k]
        # after: [batch_size, seq_len, dim_model]
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, self.dim_model)
        out = self.W_concat(out)

        return out

        




