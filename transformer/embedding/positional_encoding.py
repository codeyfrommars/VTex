import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    sinusoidal positional encoding, as described in Attention is All You Need
    """
    def __init__(self, dim_model, max_length, device):
        """
        max_length is the max sequence length of your data (gets rid of outliers)

        In addition, we apply dropout to the sums of the embeddings and the
        positional encodings in both the encoder and decoder stacks.
        """
        super(PositionalEncoding, self).__init__()
        
        self.device = device
        pe = torch.zeros(max_length, dim_model, device=self.device)

        position = torch.arange(start=0, end=max_length, dtype=torch.float, device=self.device)
        # change position from shape [max_length] into shape [max_length, 1] to allow for matrix multiplication
        position = position.unsqueeze(1)
        dimension = torch.arange(start=0, end=dim_model, step=2, dtype=torch.float, device=self.device)
        div_term = 1.0 / (10000.0 ** (dimension / dim_model))
        # change div_term from shape [dim_model] to [1, dim_model]
        div_term = div_term.unsqueeze(0)

        # phase [max_len, dim_model]
        phase = torch.matmul(position, div_term)
        assert (phase.size() == (max_length, dim_model//2)), "positional encoding not correct shape"

        pe[:, 0::2] = torch.sin(phase)
        pe[:, 1::2] = torch.cos(phase)
        self.register_buffer("pe", pe)

    def forward(self, seq_len):
        """
        Return positional encoding given SEQ_LEN
        2D tensor
        """
        return self.pe[:seq_len, :]



