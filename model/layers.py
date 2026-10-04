"""Task 2.3, 2.5, 2.6: position-wise feed-forward, encoder layer, decoder layer.

Spec:
  - FFN(x) = max(0, x W1 + b1) W2 + b2, with d_ff = 1024.
  - Encoder layer: self-attention -> add & norm -> FFN -> add & norm.
  - Decoder layer: masked self-attention -> add & norm
                   -> cross-attention over encoder output -> add & norm
                   -> FFN -> add & norm.
  - Post-norm: LayerNorm(x + Sublayer(x)), as in the paper. Dropout = 0.1.

Forbidden: nn.TransformerEncoderLayer, nn.TransformerDecoderLayer.
"""

import torch.nn as nn
from model.attention import MultiHeadAttention

class EncoderLayer(nn.Module):

    def __init__(self, d_model = 256, h = 4, d_ff = 1024, dropout = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, h)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, source_mask):
        attended, _ = self.self_attention(x , x, x, source_mask)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model = 256, h = 4, d_ff = 1024, dropout = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, h)
        self.cross_attention = MultiHeadAttention(d_model, h)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, encoder_output, target_mask, source_mask):
        attended, _ = self.self_attention(y, y, y, target_mask)
        y = self.norm1(y + self.dropout(attended))
        attended, cross_weights = self.cross_attention(y, encoder_output, encoder_output, source_mask)
        y = self.norm2(y + self.dropout(attended))
        y = self.norm3(y + self.dropout(self.feed_forward(y)))
        return y, cross_weights


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model = 256, d_ff = 1024):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
      
        return self.linear2(self.relu(self.linear1(x)))

     