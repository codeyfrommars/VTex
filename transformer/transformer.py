import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_layers = 6
    vocab_size = 10
    dim_model = 512
    num_heads = 8
    dim_ff = 2048
    dropout = 0.1
    max_length = 500
    model = Decoder(num_layers, vocab_size, dim_model, num_heads, dim_ff, dropout, max_length).to(device)

    out = model()
    print(out.shape)