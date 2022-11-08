import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    Query:
    Key:
    Value:
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

        K_T = K.transpose(2,3)

        attention = (torch.matmul(Q,  K_T)) / (dim_k ** (1/2))

        # apply optional mask (-inf)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.matmul(self.softmax(attention), V)

        return attention

        

