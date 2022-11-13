import torch
import torch.nn as nn

from decoder.decoder_layer import DecoderLayer
from embedding.transformer_embedding import DecoderEmbedding

class Decoder(nn.Module):
    """
    Decoder for VTex transformer. Target is a LaTeX string.
    """
    def __init__(self, num_layers, vocab_size, dim_model, num_heads, dim_ff, dropout, max_length):
        """
        num_layers: number of time decoder layer is repeated (N)
        vocab_size: size of dictionary
        dim_model: model dimension
        num_heads: number of heads in multi-headed attention
        dim_ff: number of hidden layers in feed forward network
        dropout: dropout probability
        max_length: max sequence length
        """
        super(Decoder, self).__init__()
        self.embed = DecoderEmbedding(dim_model, vocab_size, max_length, dropout)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(dim_model, num_heads, dim_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, vocab_size)

    def forward(self, trg, enc_out, trg_mask, enc_mask=None):
        """
        trg [batch_size, seq_len]
        enc_out [batch_size, seq_len, dim_model]
        mask [batch_size, 1, seq_len, seq_len]
        """
        # add word + position embedding
        trg = self.embed(trg)

        # apply decoder layers
        for layer in self.layers:
            trg = layer(trg, enc_out, trg_mask, enc_mask)

        # final linear layer
        out = self.linear(trg)

        return out
