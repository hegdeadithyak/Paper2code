# Attention Is All You Need

> Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.

The paper that made RNNs unemployed. Every LLM you use today is basically a scaled-up version of the right half of this diagram.

<p align="center">
  <img src="./transformer.png" alt="Full Transformer architecture" width="420" />
</p>

<sub><i>Image: Wikimedia Commons, CC BY 4.0.</i></sub>

## The only equation that matters

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

That's it. The rest of the paper (encoder/decoder stacks, positional encodings, LayerNorm, residuals, FFNs) is plumbing. This one equation is the reason we're all here.

## First principles

You have a sequence of `T` token vectors. For each position, you want to produce a new vector that's a **weighted average of all the other positions**, where the weights depend on content, not position.

- **Q** (query): "what am I looking for?" — one vector per position you're computing.
- **K** (key): "what do I offer?" — one vector per position you can look at.
- **V** (value): "what do I return if picked?" — same positions as K.

Compute `Q @ Kᵀ` → a `[T, T]` matrix of "how much position i wants position j." Softmax each row → proper weights. Multiply by `V` → each output position is a content-weighted mix of all input positions.

**Why divide by √d_k?** Because `Q @ Kᵀ` is a sum of `d_k` independent-ish products, so its variance grows with `d_k`. Plug a high-variance vector into softmax and it saturates to one-hot — gradients die. Dividing by √d_k keeps the variance ~constant regardless of dimension. The test `test_scaling_factor_matters` shows this empirically.

**Multi-head** is "do the above `h` times with different projections in parallel, then concat." Each head can specialize — one head tracks syntax, another tracks coreference, another tracks who knows what. The paper found `h=8` worked best; almost everyone has copied that since.

## Files

| File | What |
|---|---|
| `attention_scratch.py` | `scaled_dot_product_attention` (the one equation) + `MultiHeadAttentionScratch` done by hand with explicit reshape/transpose for the head split. |
| `attention_library.py` | `torch.nn.MultiheadAttention`. |
| `test_attention.py` | 10 tests: SDPA matches `F.scaled_dot_product_attention`, MHA matches `nn.MultiheadAttention` across 4 shape configs, causal mask correctness, self- vs. cross-attention, scaling-factor sanity. |

Weight layout matches PyTorch's (`in_proj_weight` stacks Q/K/V as `[3E, E]`) so you can `load_from_torch_mha` and verify bit-for-bit equivalence.

## Run it

```bash
python3 attention_scratch.py
python3 attention_library.py
python3 -m pytest test_attention.py -v -p no:anyio
```

## What to notice

- Scratch MHA matches `nn.MultiheadAttention` to **~1e-5** when weights are shared. Same math.
- The "multi-head" thing is mostly a reshape trick — one `[E, E]` matmul + a `view` into `[H, d_k]` is cheaper than `H` separate `[d_k, d_k]` matmuls.
- Causal masking is literally setting the upper triangle of the attention logits to `-inf` before the softmax. That's the whole GPT secret.

## References

- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — still the best friendly explainer.
