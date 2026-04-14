"""
Scratch ViT must match library ViT when weights are shared.
"""

import pytest
import torch

from vit_scratch import ViTScratch
from vit_library import ViTLibrary


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _share_weights(sc: ViTScratch, lib: ViTLibrary):
    sc.patch_weight = lib.patch_embed.weight.detach().clone()
    sc.patch_bias = lib.patch_embed.bias.detach().clone()
    sc.cls_token = lib.cls_token.detach().clone()
    sc.pos_embed = lib.pos_embed.detach().clone()
    for i, blk in enumerate(lib.blocks):
        p = sc.blocks[i]
        p["ln1_w"] = blk.ln1.weight.detach().clone()
        p["ln1_b"] = blk.ln1.bias.detach().clone()
        p["qkv_w"] = blk.attn.in_proj_weight.detach().clone()
        p["qkv_b"] = blk.attn.in_proj_bias.detach().clone()
        p["proj_w"] = blk.attn.out_proj.weight.detach().clone()
        p["proj_b"] = blk.attn.out_proj.bias.detach().clone()
        p["ln2_w"] = blk.ln2.weight.detach().clone()
        p["ln2_b"] = blk.ln2.bias.detach().clone()
        p["fc1_w"] = blk.mlp[0].weight.detach().clone()
        p["fc1_b"] = blk.mlp[0].bias.detach().clone()
        p["fc2_w"] = blk.mlp[2].weight.detach().clone()
        p["fc2_b"] = blk.mlp[2].bias.detach().clone()
    sc.ln_final_w = lib.ln_final.weight.detach().clone()
    sc.ln_final_b = lib.ln_final.bias.detach().clone()
    sc.head_w = lib.head.weight.detach().clone()
    sc.head_b = lib.head.bias.detach().clone()


@pytest.mark.parametrize("img,patch,E,depth,H", [
    (32, 4, 64, 2, 4),
    (16, 4, 32, 1, 4),
    (24, 8, 48, 3, 6),
])
def test_forward_matches_library(img, patch, E, depth, H):
    lib = ViTLibrary(image_size=img, patch_size=patch, embed_dim=E,
                     depth=depth, num_heads=H, num_classes=5)
    sc = ViTScratch(image_size=img, patch_size=patch, embed_dim=E,
                    depth=depth, num_heads=H, num_classes=5)
    _share_weights(sc, lib)

    x = torch.randn(2, 3, img, img)
    with torch.no_grad():
        lib_logits = lib(x)
    sc_logits = sc.forward(x)
    assert sc_logits.shape == lib_logits.shape
    assert torch.allclose(sc_logits, lib_logits, atol=1e-4, rtol=1e-3)


def test_output_shape():
    sc = ViTScratch(num_classes=100)
    x = torch.randn(4, 3, 32, 32)
    assert sc.forward(x).shape == (4, 100)


def test_patch_count():
    sc = ViTScratch(image_size=32, patch_size=4)
    assert sc.n_patches == 64
    assert sc.n_tokens == 65  # +1 for cls


def test_deterministic_for_same_input():
    sc = ViTScratch()
    x = torch.randn(2, 3, 32, 32)
    a = sc.forward(x)
    b = sc.forward(x)
    assert torch.allclose(a, b)


def test_different_batch_sizes():
    sc = ViTScratch()
    for B in [1, 3, 8]:
        out = sc.forward(torch.randn(B, 3, 32, 32))
        assert out.shape[0] == B
