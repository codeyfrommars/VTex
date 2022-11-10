import torch
import torch.nn as nn

from sublayers.multi_head_attention import MultiHead
from sublayers.position_wise_feed_forward_network import FFN
from sublayers.residual import Residual

class EncoderLayer(nn.Module):
    """
    source attention, then feed forward.
    This layer is repeated 6 times in the encoder in Attention is All You Need
    """
    def __init__(self, dim_model, num_heads, dim_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.src_attn = Residual(
            MultiHead(num_heads, dim_model),
            dim_model,
            dropout
        )
        self.ffn = Residual(
            FFN(dim_model, dim_ff),
            dim_model,
            dropout
        )

    def forward(self, enc_in, src_mask=None):
        """
        enc_in is the src
        src_mask is optional
        """
        x = self.src_attn(enc_in, enc_in, enc_in, src_mask)
        x = self.ffn(x)
        
        return x


