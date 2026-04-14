"""
Scratch MHA must match torch.nn.MultiheadAttention numerically when weights
are shared.
"""

import math
import pytest
import torch

from attention_scratch import MultiHeadAttentionScratch, scaled_dot_product_attention
from attention_library import MultiHeadAttentionLibrary


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _pair(E, H):
    lib = MultiHeadAttentionLibrary(E, H)
    sc = MultiHeadAttentionScratch(E, H)
    sc.load_from_torch_mha(lib.mha)
    return lib, sc


# ---------- scaled_dot_product_attention ----------

def test_sdpa_output_shape():
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 6, 8)
    V = torch.randn(2, 6, 10)
    out, attn = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2, 4, 10)
    assert attn.shape == (2, 4, 6)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_sdpa_matches_torch_reference():
    Q = torch.randn(3, 5, 8)
    K = torch.randn(3, 7, 8)
    V = torch.randn(3, 7, 8)
    ours, _ = scaled_dot_product_attention(Q, K, V)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_sdpa_causal_mask():
    T = 4
    Q = K = V = torch.randn(1, T, 8)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    _, attn = scaled_dot_product_attention(Q, K, V, mask=mask.view(1, T, T))
    # upper triangle should be zero (masked out)
    for i in range(T):
        for j in range(i + 1, T):
            assert attn[0, i, j].item() == 0.0


# ---------- Multi-head ----------

@pytest.mark.parametrize("B,T,E,H", [
    (1, 1, 4, 2),
    (2, 5, 16, 4),
    (3, 10, 32, 8),
    (4, 20, 64, 8),
])
def test_mha_matches_library(B, T, E, H):
    lib, sc = _pair(E, H)
    x = torch.randn(B, T, E)
    with torch.no_grad():
        lib_out, lib_attn = lib.forward(x, x, x)
    sc_out, sc_attn = sc.forward(x, x, x)
    assert sc_out.shape == lib_out.shape
    assert torch.allclose(sc_out, lib_out, atol=1e-5, rtol=1e-4)
    assert torch.allclose(sc_attn, lib_attn, atol=1e-5, rtol=1e-4)


def test_mha_cross_attention():
    """Query, key, value with different sequence lengths."""
    lib, sc = _pair(16, 4)
    q = torch.randn(2, 3, 16)
    kv = torch.randn(2, 7, 16)
    with torch.no_grad():
        lib_out, _ = lib.forward(q, kv, kv)
    sc_out, _ = sc.forward(q, kv, kv)
    assert torch.allclose(sc_out, lib_out, atol=1e-5, rtol=1e-4)


def test_mha_with_causal_mask():
    lib, sc = _pair(16, 4)
    T = 6
    x = torch.randn(2, T, 16)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    with torch.no_grad():
        lib_out, _ = lib.forward(x, x, x, attn_mask=mask)
    sc_out, _ = sc.forward(x, x, x, attn_mask=mask)
    assert torch.allclose(sc_out, lib_out, atol=1e-5, rtol=1e-4)


def test_scaling_factor_matters():
    """Demonstrate WHY the 1/√d_k scaling is in the paper: without it,
    variance of QKᵀ grows linearly with d, saturating the softmax.
    Scaled attention should be softer (lower max prob) than unscaled."""
    torch.manual_seed(1)
    Q = K = V = torch.randn(1, 20, 128)
    # scaled (our impl)
    _, attn_scaled = scaled_dot_product_attention(Q, K, V)
    # unscaled softmax for comparison
    logits = (Q @ K.transpose(-2, -1))
    attn_unscaled = torch.softmax(logits, dim=-1)
    # unscaled is sharper (closer to one-hot)
    assert attn_unscaled.max().item() > attn_scaled.max().item()
