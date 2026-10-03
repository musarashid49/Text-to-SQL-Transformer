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

# TODO: implement
