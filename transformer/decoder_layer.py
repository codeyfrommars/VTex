import torch
import torch.nn as nn

from sublayers.multi_head_attention import MultiHead
from sublayers.position_wise_feed_forward_network import FFN
from sublayers.residual import Residual

class DecoderLayer(nn.Module):
    """
    Self attention, source attention, then feed forward.
    This layer is repeated 6 times in the decoder in Attention is All You Need
    """
    def __init__(self, dim_model, num_heads, dim_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = Residual(
            MultiHead(num_heads, dim_model),
            dim_model,
            dropout
        )
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

    def forward(self, dec_in, enc_out, trg_mask, src_mask=None):
        """
        dec_in is the decoder output fed back into decoder
        enc_out is key and value returned from encoder
        trg_mask is to preserve autoregressive property
        src_mask is optional
        """
        x = self.self_attn(dec_in, dec_in, dec_in, trg_mask)
        x = self.src_attn(dec_in, enc_out, enc_out, src_mask)
        x = self.ffn(x)
        
        return x


