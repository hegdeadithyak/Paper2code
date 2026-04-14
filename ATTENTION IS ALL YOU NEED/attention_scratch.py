"""
Multi-head self-attention from scratch — Vaswani et al., 2017.

Scaled dot-product attention:
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

Multi-head is just that, done h times in parallel with different projections,
then concatenated and projected again:
    head_i = Attention(Q Wq_i, K Wk_i, V Wv_i)
    MHA(Q,K,V) = Concat(head_1, ..., head_h) Wo

Instead of h separate h×(d/h) projection matrices, we use one big (d × d)
matrix and reshape — mathematically identical, one matmul instead of h.

Weight layout matches torch.nn.MultiheadAttention so weights are swappable:
    in_proj_weight: [3*embed_dim, embed_dim]  stacked [Q; K; V]
    in_proj_bias:   [3*embed_dim]
    out_proj:       [embed_dim, embed_dim]
"""

import math
import torch


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K: [..., T_q, d_k]   V: [..., T_k, d_v]
    mask: broadcastable to the attention logits. True = mask out (set to -inf).
    Returns output [..., T_q, d_v], attn weights [..., T_q, T_k].
    """
    d_k = Q.size(-1)
    logits = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        logits = logits.masked_fill(mask, float("-inf"))
    attn = torch.softmax(logits, dim=-1)
    return attn @ V, attn


class MultiHeadAttentionScratch:
    def __init__(self, embed_dim: int, num_heads: int, dtype=torch.float32):
        assert embed_dim % num_heads == 0
        self.E = embed_dim
        self.H = num_heads
        self.d_k = embed_dim // num_heads
        k = 1.0 / math.sqrt(embed_dim)
        self.in_proj_weight = (torch.rand(3 * embed_dim, embed_dim, dtype=dtype) * 2 - 1) * k
        self.in_proj_bias = torch.zeros(3 * embed_dim, dtype=dtype)
        self.out_proj_weight = (torch.rand(embed_dim, embed_dim, dtype=dtype) * 2 - 1) * k
        self.out_proj_bias = torch.zeros(embed_dim, dtype=dtype)

    def load_from_torch_mha(self, mha: torch.nn.MultiheadAttention):
        sd = mha.state_dict()
        self.in_proj_weight = sd["in_proj_weight"].detach().clone()
        self.in_proj_bias = sd["in_proj_bias"].detach().clone()
        self.out_proj_weight = sd["out_proj.weight"].detach().clone()
        self.out_proj_bias = sd["out_proj.bias"].detach().clone()

    def forward(self, query, key, value, attn_mask=None):
        """
        query: [B, T_q, E],  key: [B, T_k, E],  value: [B, T_k, E]
        attn_mask: [T_q, T_k] bool, True = mask. (Same convention as PyTorch.)
        """
        B, T_q, E = query.shape
        T_k = key.size(1)
        H, dk = self.H, self.d_k

        # [B, T, E] → [B, T, 3E] → split → [B, T, E] each
        def proj_in(x, start):
            W = self.in_proj_weight[start:start + E]
            b = self.in_proj_bias[start:start + E]
            return x @ W.t() + b

        Q = proj_in(query, 0)
        K = proj_in(key, E)
        V = proj_in(value, 2 * E)

        # [B, T, E] → [B, H, T, d_k]
        def split_heads(x):
            return x.view(x.size(0), x.size(1), H, dk).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        mask = None
        if attn_mask is not None:
            mask = attn_mask.view(1, 1, T_q, T_k) if attn_mask.dim() == 2 else attn_mask

        out, attn = scaled_dot_product_attention(Q, K, V, mask)

        # [B, H, T_q, d_k] → [B, T_q, E]
        out = out.transpose(1, 2).contiguous().view(B, T_q, E)
        out = out @ self.out_proj_weight.t() + self.out_proj_bias

        # average attn across heads (matches torch.nn.MHA's default behavior
        # when need_weights=True, average_attn_weights=True)
        return out, attn.mean(dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, E, H = 2, 5, 16, 4
    x = torch.randn(B, T, E)
    mha = MultiHeadAttentionScratch(E, H)
    out, attn = mha.forward(x, x, x)
    print(f"out: {tuple(out.shape)}   attn: {tuple(attn.shape)}")
