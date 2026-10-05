"""Task 2.4, 2.7: masks and the full encoder-decoder model.

Spec:
  - Padding mask: hides <pad> tokens (encoder self-attn, decoder cross-attn,
    decoder self-attn).
  - Causal mask: decoder position t cannot see positions > t.
  - Full model: starter InputLayer for both sides, encoder (N = 3) and
    decoder (N = 3) stacks, and a final linear layer to the vocabulary whose
    weight IS the shared embedding matrix.
  - Report the total number of trainable parameters.

Forbidden: nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder.
"""

import torch
import torch.nn as nn
from model.layers import EncoderLayer, DecoderLayer
from tokenizer import PAD_ID      # starter/ is on sys.path (see model/__init__.py)
from embeddings import TokenEmbedding, InputLayer   # starter, used unchanged

class Encoder(nn.Module):

    def __init__(self, n_layers=3, d_model=256, h=4, d_ff = 1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)]
           )

    def forward(self, x, source_mask):
        for layer in self.layers:
            x = layer(x, source_mask)
        return x

class Decoder(nn.Module):

    def __init__(self, n_layers = 3, d_model=256, h=4, d_ff= 1024, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, y, encoder_output, target_mask, source_mask):
        cross_weights = None          # defined even if there are no layers
        for layer in self.layers:
            y, cross_weights = layer(y, encoder_output, target_mask, source_mask)

        return y, cross_weights

def make_source_mask(source, pad_id=PAD_ID):
    """Padding mask: True at real tokens, False at <pad>.
    (B, source_position) -> (B, 1, 1, source_position)"""
    return (source != pad_id)[:, None, None, :]


def make_target_mask(target, pad_id=PAD_ID):
    """Padding mask AND causal mask, for decoder self-attention 
    (B, target_position) -> (B, 1, target_position, target_position)"""
    target_length = target.size(1)
    padding = (target != pad_id)[:, None, None, :]
    causal = torch.tril(torch.ones(target_length, target_length,
                                   dtype=torch.bool, device=target.device))
    return padding & causal


class Transformer(nn.Module):
    """Full encoder-decoder model (paper §3, Figure 1). One shared weight matrix
    for the encoder embedding, decoder embedding and output projection (§3.4)."""

    def __init__(self, vocab_size, d_model=256, h=4, n_layers=3, d_ff=1024,
                 dropout=0.1, max_len=512, pad_id=PAD_ID):
        super().__init__()
        self.pad_id = pad_id

        self.shared_embedding = TokenEmbedding(vocab_size, d_model, pad_id)
        # re-initialise so that embedding * sqrt(d_model) has std ~1 (same scale as PE)
        nn.init.normal_(self.shared_embedding.emb.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.shared_embedding.emb.weight[pad_id].zero_()

        self.encoder_input = InputLayer(self.shared_embedding, d_model, max_len, dropout)
        self.decoder_input = InputLayer(self.shared_embedding, d_model, max_len, dropout)
        self.encoder = Encoder(n_layers, d_model, h, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, h, d_ff, dropout)

        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.shared_embedding.emb.weight   # tie: same tensor

    def encode(self, source):
        source_mask = make_source_mask(source, self.pad_id)
        encoder_output = self.encoder(self.encoder_input(source), source_mask)
        return encoder_output, source_mask

    def decode(self, target_in, encoder_output, source_mask):
        target_mask = make_target_mask(target_in, self.pad_id)
        y, cross_weights = self.decoder(self.decoder_input(target_in),
                                        encoder_output, target_mask, source_mask)
        return self.output_projection(y), cross_weights

    def forward(self, source, target_in):
        # source: (B, source_position) ids;  target_in: (B, target_position) ids = tgt[:, :-1]
        encoder_output, source_mask = self.encode(source)
        logits, _ = self.decode(target_in, encoder_output, source_mask)
        return logits            # (B, target_position, vocab_size)
