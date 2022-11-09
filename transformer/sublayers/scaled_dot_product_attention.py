import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    """
    Scaled dot product attention model
    """
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V are same dimensions
        mask is optionally used to preserve autoregressive property
        """
        batch_size, num_heads, seq_len, dim_k = K.size()

        K_T = K.transpose(-2,-1)

        # attention is of shape [batch_size, num_heads, seq_len, seq_len]
        attention = (torch.matmul(Q,  K_T)) / math.sqrt(dim_k)

        # apply optional mask (-inf)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.matmul(self.softmax(attention), V)

        return attention

        

