"""Task 2.1 - 2.2: scaled dot-product attention and multi-head attention.

Spec (Vaswani et al., 2017, Section 3.2):
  - Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V, with an optional mask.
    Masked positions get -inf (or a very large negative number) before softmax.
    Return BOTH the output and the attention weights.
  - Multi-head: project Q, K, V with W_i^Q, W_i^K, W_i^V, run h heads in
    parallel, concatenate, and project with W^O.
  - Config: d_model = 256, h = 4 (d_k = d_v = 64).

Forbidden: nn.MultiheadAttention, F.scaled_dot_product_attention.

Mask convention (used everywhere in model/): a bool tensor broadcastable to
(B, h, query_position, key_position) where True = "may attend" and
False = "blocked".
"""

import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) /math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim = -1)
        output = torch.einsum("bhqk,bhkd->bhqd", weights, v)
        return output, weights
    
class MultiHeadAttention(nn.Module):

     def __init__(self, d_model=256, h = 4):
         super().__init__()
         assert d_model % h == 0, "Model dimensions should be evenly split across heads."
         self.h = h 
         self.d_k = d_model // h 

         self.w_q = nn.Parameter(torch.empty(h,d_model, self.d_k))
         self.w_k = nn.Parameter(torch.empty(h, d_model, self.d_k))
         self.w_v = nn.Parameter(torch.empty(h, d_model, self.d_k))
         self.b_q = nn.Parameter(torch.zeros(h,self.d_k))
         self.b_k = nn.Parameter(torch.zeros(h, self.d_k))
         self.b_v = nn.Parameter(torch.zeros(h, self.d_k))

         self.w_o = nn.Parameter(torch.empty(h, self.d_k, d_model))
         self.b_o = nn.Parameter(torch.zeros(d_model))

         bound = math.sqrt(6 / (d_model + d_model))
         for w in (self.w_q, self.w_k, self.w_v, self.w_o):
             nn.init.uniform_(w, -bound, bound)

         self.attention = ScaledDotProductAttention()

     def forward(self, query, key, value, mask=None):
         # einsum letters: b = batch, t = position, m = d_model, h = head, d = d_k
         q = torch.einsum("btm,hmd->bhtd", query, self.w_q) + self.b_q[:, None, :]
         k = torch.einsum("btm,hmd->bhtd", key, self.w_k) + self.b_k[:, None,:]
         v = torch.einsum("btm,hmd->bhtd", value, self.w_v) + self.b_v[:, None, :]

         heads, weights = self.attention(q, k, v, mask)
         output = torch.einsum("bhtd,hdm->btm", heads, self.w_o) + self.b_o
         return output, weights