import torch
import torch.nn as nn

from decoder_layer import DecoderLayer
from embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
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
        super(Decoder, self).__init__()
        self.embed = TransformerEmbedding(dim_model, vocab_size, max_length, dropout)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(dim_model, num_heads, dim_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, vocab_size)

    def make_trg_mask(self, size):
        "Mask out subsequent positions for self attention"
        mask = torch.full(
            (size, size), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, src, src_mask=None):
        # add word + position embedding
        trg = self.embed(trg)

        # build trg_mask for masked multi-head attention
        # trg is shape [batch_size, seq_len, dim_model]
        _, seq_len = trg.size()
        trg_mask = self.make_trg_mask(seq_len)

        # apply decoder layers
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # final linear layer
        out = self.linear(trg)

        return out
