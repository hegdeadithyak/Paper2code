"""
Multi-head attention via torch.nn.MultiheadAttention.
Same math, fused matmul, written in C++. Use batch_first=True to match the
[B, T, E] convention everyone actually uses in 2020s code.
"""

import torch
import torch.nn as nn


class MultiHeadAttentionLibrary:
    def __init__(self, embed_dim: int, num_heads: int):
        self.E = embed_dim
        self.H = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            bias=True,
        )

    def forward(self, query, key, value, attn_mask=None):
        out, attn = self.mha(query, key, value, attn_mask=attn_mask,
                             need_weights=True, average_attn_weights=True)
        return out, attn


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, E, H = 2, 5, 16, 4
    x = torch.randn(B, T, E)
    mha = MultiHeadAttentionLibrary(E, H)
    out, attn = mha.forward(x, x, x)
    print(f"out: {tuple(out.shape)}   attn: {tuple(attn.shape)}")
