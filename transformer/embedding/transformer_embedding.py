import torch
import torch.nn as nn

from embedding.embeddings import Embeddings
from embedding.positional_encoding import PositionalEncoding

class TransformerEmbedding(nn.Module):
    """
    Applies token embedding, adds the positional encoding, then applies dropout
    """
    def __init__(self, dim_model, vocab_size, max_length, dropout):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = Embeddings(vocab_size, dim_model)
        self.pos_emb = PositionalEncoding(dim_model, max_length)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        embeds target tensor X
        """
        tok_emb = self.tok_emb(x)
        # tok_emb is [batch_size, seq_len, dim_model]
        _, seq_len, _ = tok_emb.size()
        # pos_emb is [seq_len, dim_model]
        pos_emb = self.pos_emb(seq_len)
        # TODO: pos_emb should be properly broadcasted right?
        return self.dropout(tok_emb + pos_emb)