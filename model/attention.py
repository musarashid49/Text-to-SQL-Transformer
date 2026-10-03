"""Task 2.1 - 2.2: scaled dot-product attention and multi-head attention.

Spec (Vaswani et al., 2017, Section 3.2):
  - Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V, with an optional mask.
    Masked positions get -inf (or a very large negative number) before softmax.
    Return BOTH the output and the attention weights.
  - Multi-head: project Q, K, V with W_i^Q, W_i^K, W_i^V, run h heads in
    parallel, concatenate, and project with W^O.
  - Config: d_model = 256, h = 4 (d_k = d_v = 64).

Forbidden: nn.MultiheadAttention, F.scaled_dot_product_attention.
"""

# TODO: implement
