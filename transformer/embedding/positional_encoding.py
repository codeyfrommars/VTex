import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    sinusoidal positional encoding, as described in Attention is All You Need
    """
    def __init__(self, dim_model, max_length):
        """
        max_length is the max sequence length of your data (gets rid of outliers)

        In addition, we apply dropout to the sums of the embeddings and the
        positional encodings in both the encoder and decoder stacks.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, dim_model)

        # TODO: check this works
        position = torch.arange(max_length, dtype=torch.float)
        # change position from shape [max_length] into shape [max_length, 1] to allow for matrix multiplication
        position = position.unsqueeze(1)
        dimension = torch.arange(dim_model, step=2, dtype=torch.float)
        div_term = 1.0 / (10000.0 ** (dimension / dim_model))

        phase = torch.matmul(position, div_term)

        pe[:, 0::2] = torch.sin(phase)
        pe[:, 1::2] = torch.cos(phase)
        self.register_buffer("posenc", pe)

    def forward(self, seq_len):
        """
        Return positional encoding given SEQ_LEN
        2D tensor
        """
        return self.pe[:seq_len, :]



