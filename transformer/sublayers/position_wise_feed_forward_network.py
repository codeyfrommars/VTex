import torch
import torch.nn as nn

class FFN(nn.Module):
    """
    Position-wise feed-forward network
    """
    def __init__(self, dim_model, dim_ff):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_ff, dim_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


