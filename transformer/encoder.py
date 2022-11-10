import torch
import torch.nn as nn

from encoder_layer import EncoderLayer
from embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    """
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
        super(Encoder, self).__init__()
        self.embed = TransformerEmbedding(dim_model, vocab_size, max_length, dropout)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(dim_model, num_heads, dim_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src, src_mask=None):
        """
        src [batch_size, seq_len]
        mask [batch_size, 1, seq_len, seq_len]
        """
        # add word + position embedding
        src = self.embed(src)

        # apply encoder layers
        for layer in self.layers:
            src = layer(src, src_mask)

        return src
