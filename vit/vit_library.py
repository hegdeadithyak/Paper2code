"""
ViT via torch.nn modules — patch embed as Conv2d, encoder as TransformerEncoderLayer.
Weight names chosen so we can share them with vit_scratch.
"""

import torch
import torch.nn as nn


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.mlp(self.ln2(x))


class ViTLibrary(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=64,
                 depth=2, num_heads=4, mlp_ratio=4, num_classes=10):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.n_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        h = self.patch_embed(x).flatten(2).transpose(1, 2)   # [B, N, E]
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos_embed
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_final(h)[:, 0])


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ViTLibrary()
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    print(f"input: {tuple(x.shape)}   logits: {tuple(logits.shape)}")
