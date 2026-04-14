# Vision Transformer (ViT)

> Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.

The paper that looked at CNNs and said "lol, we don't need inductive biases if we have enough data." Chop the image into patches, treat each patch as a word, feed to a vanilla Transformer, hit state-of-the-art on ImageNet when you have 300M images to train on.

<p align="center">
  <img src="./vit.gif" alt="Vision Transformer architecture" width="520" />
</p>

<sub><i>Image: Wikimedia Commons, CC BY 4.0.</i></sub>

## First principles

Transformers work on sequences of tokens. Images aren't sequences — they're 2D grids. Previous attempts to transformer-ify images fed pixels one at a time (`n=224×224=50176` tokens → O(n²) attention → you can't afford it). The ViT trick is cheap:

1. **Patchify.** Split the image into `P×P` patches (usually 16×16). A 224×224 image becomes 14×14 = 196 patches, each a `16×16×3=768`-dim vector. This is literally `Conv2d(kernel=P, stride=P)`.
2. **Linear project** each patch to `embed_dim`. Now it's a sequence of `N` token embeddings.
3. **Prepend a `[CLS]` token.** A learnable vector that accumulates info from all patches and gets classified at the end. Borrowed wholesale from BERT.
4. **Add learnable positional embeddings.** The transformer has zero spatial prior; the position embedding is the *only* way it knows that patch 7 is next to patch 8.
5. **Run through `depth` transformer encoder blocks** (pre-norm: LN → attention → residual → LN → MLP → residual).
6. **Read the CLS token from the final layer**, put it through a linear classifier. Done.

No convolutions (after patchify). No pooling. No pyramids. No anchors. Just attention all the way down.

## Why it works (and when it doesn't)

ViT *underperforms* a comparable ResNet on ImageNet-1k when trained from scratch. The CNN's translation equivariance and local-receptive-field prior are genuinely useful when data is limited. But feed ViT a dataset of JFT-300M images and it overtakes ResNet, because at that scale the model can *learn* the priors rather than being forced into them.

Lesson: inductive biases are a data-efficiency tradeoff. When data is infinite, they cost you.

## Files

| File | What |
|---|---|
| `vit_scratch.py` | Every op by hand: patchify (conv2d), manual MHA, manual LayerNorm, manual MLP, assembled into a `depth`-deep encoder with CLS token + pos embed + classifier head. No `nn.Module`, parameters held as plain tensors. |
| `vit_library.py` | Same model built from `nn.Linear`, `nn.LayerNorm`, `nn.MultiheadAttention`, `nn.Conv2d`. |
| `test_vit.py` | 7 tests: forward parity with library across 3 configs (different image/patch/embed/depth/heads), shape/determinism sanity. |

Default config: 32×32 input, 4×4 patches, embed=64, depth=2, heads=4, 10 classes. Small enough that tests run in 2s.

## Run it

```bash
python3 vit_scratch.py
python3 vit_library.py
python3 -m pytest test_vit.py -v -p no:anyio
```

## What to notice

- Matches library to **~1e-4** across 3 distinct configs. LayerNorm accumulates more float error than MHA alone, so tolerance is looser than for LSTM/MHA.
- The "patchification" step really is just a `conv2d` with `stride=patch_size`. There's no separate "flatten each patch" code — the convolution does it implicitly. Satisfying.
- The CLS token is initialized to *zero* and learned from scratch. For the first several training steps it carries no information; pos embeddings and MLP weights do all the lifting, then it gradually becomes a useful "summary slot."
- Pre-norm (LayerNorm *before* attention/MLP) matters for stable training at depth — the original paper used it; earlier transformer variants used post-norm and were finicky past depth 6.

## References

- Dosovitskiy et al. — [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Touvron et al. — [Training data-efficient image transformers & distillation (DeiT)](https://arxiv.org/abs/2012.12877) — how to train ViT on ImageNet-1k without JFT-300M.
