import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    """
    embedding to convert input/output tokens to vectors of dimensions DIM_MODEL
    """
    def __init__(self, vocab_size, dim_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, dim_model)
        self.norm = nn.LayerNorm(dim_model)
        self.dim_model = dim_model

    def forward(self, x):
        """
        3D tensor
        """
        # return self.lut(x) * math.sqrt(self.dim_model)
        out = self.lut(x)
        out = self.norm(out)
        return out