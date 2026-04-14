"""
Vision Transformer from scratch — Dosovitskiy et al., 2020,
*An Image is Worth 16x16 Words*.

Core idea: if transformers work on sequences of tokens, an image is just a
sequence of patch tokens. Chop the image into 16×16 patches, flatten each to
a vector, project to embed_dim, prepend a [CLS] token, add learnable positional
embeddings, run through a stack of transformer encoder blocks, read off the
CLS token, classify.

That's it. No convolutions (outside the patchify step, which is literally
a conv2d with stride=patch_size), no hand-crafted vision priors. It beats
CNNs when given enough data.

This file uses pure tensor ops — no nn.Module, no nn.MultiheadAttention, no
nn.LayerNorm. Weight layout is chosen so we can copy weights to/from a
library-nn.Module-based reference.
"""

import math
import torch


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def gelu(x):
    # tanh approximation — matches nn.GELU(approximate='tanh').
    # For exact-match with nn.GELU() default, use torch.nn.functional.gelu.
    return torch.nn.functional.gelu(x)


def multi_head_attention(x, qkv_w, qkv_b, proj_w, proj_b, num_heads):
    """
    x: [B, N, E]
    qkv_w: [3E, E], qkv_b: [3E]
    proj_w: [E, E], proj_b: [E]
    """
    B, N, E = x.shape
    dk = E // num_heads

    qkv = x @ qkv_w.t() + qkv_b                       # [B, N, 3E]
    qkv = qkv.view(B, N, 3, num_heads, dk).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]                  # [B, H, N, dk]

    attn = (q @ k.transpose(-2, -1)) / math.sqrt(dk)  # [B, H, N, N]
    attn = torch.softmax(attn, dim=-1)
    out = attn @ v                                    # [B, H, N, dk]
    out = out.transpose(1, 2).contiguous().view(B, N, E)
    return out @ proj_w.t() + proj_b


def transformer_block(x, block_params, num_heads):
    """Pre-norm transformer block (what ViT uses)."""
    p = block_params
    # attn sub-layer
    h = layer_norm(x, p["ln1_w"], p["ln1_b"])
    h = multi_head_attention(h, p["qkv_w"], p["qkv_b"], p["proj_w"], p["proj_b"], num_heads)
    x = x + h
    # MLP sub-layer
    h = layer_norm(x, p["ln2_w"], p["ln2_b"])
    h = gelu(h @ p["fc1_w"].t() + p["fc1_b"])
    h = h @ p["fc2_w"].t() + p["fc2_b"]
    return x + h


class ViTScratch:
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=64,
                 depth=2, num_heads=4, mlp_ratio=4, num_classes=10, dtype=torch.float32):
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.n_patches = (image_size // patch_size) ** 2
        self.n_tokens = self.n_patches + 1  # + cls

        k = 1.0 / math.sqrt(embed_dim)
        mlp_hidden = mlp_ratio * embed_dim

        # Patch embedding as a conv2d with stride=patch_size.
        # patch_weight: [E, C, P, P]
        self.patch_weight = (torch.randn(embed_dim, in_channels, patch_size, patch_size, dtype=dtype)) * k
        self.patch_bias = torch.zeros(embed_dim, dtype=dtype)

        self.cls_token = torch.zeros(1, 1, embed_dim, dtype=dtype)
        self.pos_embed = torch.zeros(1, self.n_tokens, embed_dim, dtype=dtype)

        def new_block():
            return {
                "ln1_w": torch.ones(embed_dim, dtype=dtype),
                "ln1_b": torch.zeros(embed_dim, dtype=dtype),
                "qkv_w": (torch.randn(3 * embed_dim, embed_dim, dtype=dtype)) * k,
                "qkv_b": torch.zeros(3 * embed_dim, dtype=dtype),
                "proj_w": (torch.randn(embed_dim, embed_dim, dtype=dtype)) * k,
                "proj_b": torch.zeros(embed_dim, dtype=dtype),
                "ln2_w": torch.ones(embed_dim, dtype=dtype),
                "ln2_b": torch.zeros(embed_dim, dtype=dtype),
                "fc1_w": (torch.randn(mlp_hidden, embed_dim, dtype=dtype)) * k,
                "fc1_b": torch.zeros(mlp_hidden, dtype=dtype),
                "fc2_w": (torch.randn(embed_dim, mlp_hidden, dtype=dtype)) * (1.0 / math.sqrt(mlp_hidden)),
                "fc2_b": torch.zeros(embed_dim, dtype=dtype),
            }

        self.blocks = [new_block() for _ in range(depth)]

        self.ln_final_w = torch.ones(embed_dim, dtype=dtype)
        self.ln_final_b = torch.zeros(embed_dim, dtype=dtype)
        self.head_w = (torch.randn(num_classes, embed_dim, dtype=dtype)) * k
        self.head_b = torch.zeros(num_classes, dtype=dtype)

    def forward(self, x: torch.Tensor):
        """x: [B, C, H, W] -> logits [B, num_classes]"""
        B = x.size(0)
        # patchify via conv
        h = torch.nn.functional.conv2d(x, self.patch_weight, self.patch_bias,
                                       stride=self.patch_size)   # [B, E, H/P, W/P]
        h = h.flatten(2).transpose(1, 2)                         # [B, N, E]

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)                           # [B, N+1, E]
        h = h + self.pos_embed

        for block in self.blocks:
            h = transformer_block(h, block, self.num_heads)

        h = layer_norm(h, self.ln_final_w, self.ln_final_b)
        cls_out = h[:, 0]                                        # [B, E]
        return cls_out @ self.head_w.t() + self.head_b           # [B, num_classes]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ViTScratch(image_size=32, patch_size=4, embed_dim=64,
                       depth=2, num_heads=4, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    logits = model.forward(x)
    print(f"input: {tuple(x.shape)}   logits: {tuple(logits.shape)}")
