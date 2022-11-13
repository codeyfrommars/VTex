import torch
import torch.nn as nn

from encoder.encoder_classic import Encoder
from decoder.decoder import Decoder

class Transformer(nn.Module):
    """
    Encoder-Decoder transformer
    """
    def __init__(
            self,
            device,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            max_length,
            num_layers=6,
            num_heads=8,
            dim_model=512,
            dim_ff=2048,
            dropout=0.1
        ):
        super(Transformer, self).__init__()
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        
        self.encoder = Encoder(num_layers, src_vocab_size, dim_model, num_heads, dim_ff, dropout, max_length)
        self.decoder = Decoder(num_layers, trg_vocab_size, dim_model, num_heads, dim_ff, dropout, max_length)

    def forward(self, src, trg):
        """
        src [batch_size, src_seq_len]
        trg [batch_size, trg_seq_len]
        """
        src_mask = self._make_pad_mask(src, self.src_pad_idx)
        # TODO: encoder-decoder pad mask
        # enc_dec_mask = self.make_pad_mask()
        trg_mask = self._make_pad_mask(trg, self.trg_pad_idx) * self._make_trg_mask(trg)

        # enc_out [batch_size, src_seq_len, dim_model]
        enc_out = self.encoder(src, src_mask)
        # dec_out [batch_size, trg_seq_len, trg_vocab_size]
        dec_out = self.decoder(trg, enc_out, trg_mask)
        
        return dec_out

    
    def _make_pad_mask(self, x, pad):
        "This mask hides padding"
        # x is shape [batch_size, seq_len]
        batch_size, seq_len = x.size()
        # mask [batch_size, 1, 1, seq_len]
        mask = torch.ne(x, pad).unsqueeze(1).unsqueeze(2)
        # mask [batch_size, 1, seq_len, seq_len]
        mask = mask.repeat(1, 1, seq_len, 1)

        assert (mask.size() == (batch_size, 1, seq_len, seq_len)), "make_pad_mask incorrect"
        return mask
    
    def _make_trg_mask(self, trg):
        """
        This mask hides future words, preserving autoregressive property
        [[1, 0, 0]
         [1, 1, 0]
         [1, 1, 1]]
        Returns [seq_len, seq_len]
        """
        # trg is shape [batch_size, seq_len]
        _, seq_len = trg.size()
        mask = torch.full(
            (seq_len, seq_len), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(diagonal=1)  # zero out the upper diagonal
        # mask [seq_len, seq_len]
        assert (mask.size() == (seq_len, seq_len)), "make_trg_mask incorrect"
        return mask == 0


if __name__ == "__main__":
    """
    example code to verify functionality of Transformer
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src = torch.tensor([[1,5,6,4,3,4,7,2,0],[1,5,3,6,7,1,9,9,2]]).to(device)
    trg = torch.tensor([[1,7,3,4,7,2,0],[1,4,3,5,7,9,2]]).to(device)

    src_vocab_size = 10
    trg_vocab_size = 10
    src_pad_idx = 0
    trg_pad_idx = 0
    max_length=100
    num_layers=6
    num_heads=8
    dim_model=512
    dim_ff=2048
    dropout=0.1

    model = Transformer(device, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, max_length).to(device)

    out = model(src, trg[:, :-1])
    print(out.shape)