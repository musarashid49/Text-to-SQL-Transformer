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

# TODO: implement
