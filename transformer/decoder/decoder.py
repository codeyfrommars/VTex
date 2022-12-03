import torch
import torch.nn as nn

from decoder.decoder_layer import DecoderLayer
from embedding.transformer_embedding import DecoderEmbedding

class Decoder(nn.Module):
    """
    Decoder for VTex transformer. Target is a LaTeX string.
    """
    def __init__(self, num_layers, vocab_size, dim_model, num_heads, dim_ff, dropout, max_length, device):
        """
        num_layers: number of time decoder layer is repeated (N)
        vocab_size: size of dictionary
        dim_model: model dimension
        num_heads: number of heads in multi-headed attention
        dim_ff: number of hidden layers in feed forward network
        dropout: dropout probability
        max_length: max sequence length
        """
        super(Decoder, self).__init__()
        self.device = device
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embed = DecoderEmbedding(dim_model, vocab_size, max_length, dropout, device=device)

        # self.layers = nn.ModuleList(
        #     [
        #         DecoderLayer(dim_model, num_heads, dim_ff, dropout)
        #         for _ in range(num_layers)
        #     ]
        # )
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.linear = nn.Linear(dim_model, vocab_size)

    def forward(self, trg, enc_out, trg_mask, pad_idx, enc_mask=None):
        """
        trg [batch_size, seq_len]
        enc_out [batch_size, seq_len, dim_model]
        mask [batch_size, 1, seq_len, seq_len]
        """
        # Create pad mask
        trg_pad_mask = trg == pad_idx

        # add word + position embedding
        trg = self.embed(trg) # [b, l, dim]

        # # apply decoder layers
        # for layer in self.layers:
        #     trg = layer(trg, enc_out, trg_mask, enc_mask)

        # Reshape to [len, batch, dim]
        enc_out = enc_out.permute(1,0,2)
        trg = trg.permute(1,0,2)
        trg = self.decoder(tgt=trg, memory=enc_out, tgt_mask=trg_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=enc_mask)

        # Reshape to [batch, len, dim]
        trg = trg.permute(1,0,2)
        # final linear layer
        out = self.linear(trg)

        return out

    def beam_search(self, enc_out, pad_idx, sos_idx, eos_idx, beam_size):
        """
        Performs beam search for best hypothesis
        enc_out [1, seq_len, dim_model]
        Return output index sequence [seq_len], includes SOS and EOS
        """

        hypotheses = torch.full(
            (1, self.max_length),
            fill_value=pad_idx,
            dtype=torch.long,
            device=self.device,
        )
        assert (enc_out.size(0) == 1), f"beam search should only have single source, encounter with batch_size: {enc_out.size(0)}"

        hypotheses[:, 0] = sos_idx

        hyp_scores = torch.zeros(1, dtype=torch.float, device=self.device)

        t = 0
        while t < self.max_length-1:
            hyp_num = hypotheses.size(0)
            assert hyp_num <= beam_size, f"hyp_num: {hyp_num}, beam_size: {beam_size}"
            # Decoder prediction
            enc_out_reshape = enc_out.repeat(hyp_num, 1, 1) # [hyp_num, seq_len, dim_model]
            mask = self._make_pad_mask(hypotheses, pad_idx)
            decoder_output = self(hypotheses, enc_out_reshape, mask) # [hyp_num, seq_len, classes]
            decoder_output = decoder_output[:, t, :] # take the seq_len relevant to us
            log_probs = nn.functional.log_softmax(decoder_output, dim=-1) #[hyp_num, classes]
            log_probs = log_probs / self._sequence_length_penalty(t+1, 0.6) # penalize longer sequences (optional)

            # Set score to zero where EOS/PAD has been reached
            log_probs[hypotheses[:, t]==eos_idx, :] = 0
            log_probs[hypotheses[:, t]==pad_idx, :] = 0

            # Get top predictions
            hyp_scores = hyp_scores.unsqueeze(1) + log_probs # [hyp_num, classes]
            hyp_scores, indices = torch.topk(hyp_scores.reshape(-1), beam_size)
            beam_indices  = torch.divide(indices, self.vocab_size, rounding_mode='floor') # indices // vocab_size
            token_indices = torch.remainder(indices, self.vocab_size)                     # indices %  vocab_size

            # Create new hypothesis
            t = t+1
            new_hypotheses = []
            for beam_index, token_index in zip(beam_indices, token_indices):
                prev_hypothesis = hypotheses[beam_index]
                if prev_hypothesis[t-1]==eos_idx or prev_hypothesis[t-1]==pad_idx: # do not change completed hypotheses
                    token_index = pad_idx
                hypotheses[beam_index,t] = token_index
                new_hypotheses.append(hypotheses[beam_index].detach().clone())
            hypotheses = torch.stack(new_hypotheses, dim=0)

            # If all beams finished, exit
            if (hypotheses[:, t]==eos_idx).sum() + (hypotheses[:,t]==pad_idx).sum() == beam_size:
                break
        
        # Get the top scored hypothesis
        best_hyp, _ = max(zip(hypotheses, hyp_scores), key=lambda x: x[1])

        # Remove pads
        ret_hyp = best_hyp[best_hyp!=pad_idx]
        return ret_hyp


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

    def _sequence_length_penalty(self, length, alpha: float=0.6) -> float:
        return ((5 + length) / (5 + 1)) ** alpha






