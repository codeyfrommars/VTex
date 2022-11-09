import torch
import torch.nn as nn
from torch import Tensor

class Residual(nn.Module):
    """
    The Add & Norm layer in Attention is All You Need
    """
    def __init__(self, sublayer: nn.Module, dim_model, dropout):
        super(Residual, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, *tensors: Tensor) -> Tensor:
        """
        We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
        tensors order for MultiHead is Q, K, V, mask
        """
        # tensors[0] is Q according to MultiHead signature
        return self.norm(self.dropout(self.sublayer(*tensors)) + tensors[0])

