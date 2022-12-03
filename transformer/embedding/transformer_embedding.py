import torch
import torch.nn as nn

from embedding.embeddings import Embeddings
from embedding.positional_encoding import PositionalEncoding

class DecoderEmbedding(nn.Module):
    """
    Applies token embedding, adds the positional encoding, then applies dropout
    """
    def __init__(self, dim_model, vocab_size, max_length, dropout, device):
        super(DecoderEmbedding, self).__init__()
        self.device = device
        self.tok_emb = Embeddings(vocab_size, dim_model)
        self.pos_emb = PositionalEncoding(dim_model, max_length, self.device)
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
        
        return self.dropout(tok_emb + pos_emb)

class EncoderEmbedding(nn.Module):
    """
    Applies positional encoding over 2D, embedding pixel position information
    """
    def __init__(self, dim_model, device):
        super(EncoderEmbedding, self).__init__()
        # [width, dim_model//2]
        # self.x_pos_emb = PositionalEncoding(dim_model//2, width)
        # [height, dim_model//2]
        # self.y_pos_emb = PositionalEncoding(dim_model//2, height)
        self.device = device
        self.dim_model = dim_model

    def forward(self, x):
        """
        Adds the positional encoding to x
        x [batch_size, height, width, dim_model]
        """
        _, height, width, _ = x.size()
        x_pos = PositionalEncoding(self.dim_model//2, width, device=self.device)
        y_pos = PositionalEncoding(self.dim_model//2, height, device=self.device)
        # Concatenate the x_pos_emb and y_pos_emb
        x_pos_emb = x_pos(width) 
        y_pos_emb = y_pos(height)
        # [width, dim_model//2] -> [height, width, dim_model//2]
        x_pos_emb = x_pos_emb.unsqueeze(0)
        x_pos_emb = x_pos_emb.repeat(height, 1, 1)
        # [height, dim_model//2] -> [height, width, dim_model//2]
        y_pos_emb = y_pos_emb.unsqueeze(1)
        y_pos_emb = y_pos_emb.repeat(1, width, 1)
        assert (x_pos_emb.size() == y_pos_emb.size() == (height, width, self.dim_model//2)), "x,y positional encodings incorrect size"
    
        # [height, width, dim_model]
        pos_emb = torch.cat((x_pos_emb, y_pos_emb), dim=2)

        assert (pos_emb.size() == (height, width, self.dim_model)), "image positional encodings incorrect size"

        return x + pos_emb

