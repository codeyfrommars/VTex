import torch
import torch.nn as nn
import train

from encoder.encoder import Encoder
from decoder.decoder import Decoder

class Transformer(nn.Module):
    """
    Encoder-Decoder transformer
    """
    def __init__(
            self,
            device,
            trg_vocab_size,
            trg_pad_idx,
            max_trg_length,
            img_height,
            img_width,
            growth_rate=24,
            block_depth=16,
            compression=0.5,
            num_layers=6,
            num_heads=8,
            dim_model=512,
            dim_ff=2048,
            dropout_enc=0.2,
            dropout_dec=0.1
        ):
        super(Transformer, self).__init__()
        self.device = device

        self.trg_pad_idx = trg_pad_idx
        self.height = img_height
        self.width = img_width
        
        self.encoder = Encoder(growth_rate, block_depth, compression, dropout_enc, img_height, img_width, dim_model)
        self.decoder = Decoder(num_layers, trg_vocab_size, dim_model, num_heads, dim_ff, dropout_dec, max_trg_length)

    def forward(self, src, trg):
        """
        src [batch_size, channels=1, height, width]
        trg [batch_size, trg_seq_len]
        """

        # Create pad masks for transformer
        # TODO: encoder-decoder pad mask
        # enc_dec_mask = self.make_pad_mask()
        trg_mask = self._make_pad_mask(trg, self.trg_pad_idx) * self._make_trg_mask(trg)

        # Encoder
        # enc_out [batch_size, (height//16) * (width//16), dim_model]
        enc_out = self.encoder(src)

        # Decoder
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

    # new src [2, 1, 1000, 1000]
    src1 = torch.rand(1000, 1000).unsqueeze(0).to(device)
    src2 = torch.rand(1000, 1000).unsqueeze(0).to(device)
    src = torch.stack((src1, src2), dim=0)

    trg = torch.tensor([[1,7,3,4,7,2,0],[1,4,3,5,7,9,2]]).to(device)

    trg_vocab_size = 10
    trg_pad_idx = 0 # What index in the dictory is the pad character, 2 is EOS character
    max_trg_length = 100
    img_height = 1000
    img_width = 1000

    model = Transformer(device, trg_vocab_size, trg_pad_idx, max_trg_length, img_height, img_width).to(device)

    out = model(src, trg[:, :-1])
    print(out.shape)

    train(model, device)