# RNN Encoder-Decoder with GRU

> Cho et al., *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, EMNLP 2014.

Folder name says "Convolutional RNN" but the paper that matters here is Cho 2014 — the one that **invented the seq2seq encoder-decoder** and introduced the **GRU** along the way. The Transformer ate its lunch three years later, but every attention-based seq-to-seq model in existence owes this paper its framing.

<p align="center">
  <img src="./gru_cell.svg" alt="GRU cell" width="500" />
</p>

<sub><i>Image: Wikimedia Commons, CC BY-SA 4.0.</i></sub>

## GRU — LSTM on a diet

LSTM has four gates and two states (`h`, `c`). GRU merges them:
- Drops the separate cell state; there's just `h`.
- Merges input + forget gate into one **update gate** `z`: "how much of the old `h` do I keep vs. replace?"
- Keeps a **reset gate** `r`: "when computing the new candidate, how much of the old `h` am I even allowed to look at?"

Two gates. One state. ~25% fewer parameters than LSTM. In practice the two perform similarly on most tasks, which is slightly embarrassing for LSTM.

## Equations

```
r_t = σ(x_t W_ir + h_{t-1} W_hr + b_r)                    reset
z_t = σ(x_t W_iz + h_{t-1} W_hz + b_z)                    update
n_t = tanh(x_t W_in + r_t ⊙ (h_{t-1} W_hn + b_hn) + b_in) candidate
h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}                     interpolate
```

The update line is the cute bit: `h` is literally a convex combination of "what it was" and "what the candidate suggests." When `z ≈ 1` the cell is frozen and gradients sail through. That's GRU's version of LSTM's gradient-highway trick.

## Files

| File | What |
|---|---|
| `gru_scratch.py` | Manual gate equations, tensor ops only. Weight layout matches `torch.nn.GRU` (`[3H, I]`, stacked `[r, z, n]`). |
| `gru_library.py` | `torch.nn.GRU` wrapper. |
| `test_gru.py` | 8 tests: forward shapes, match `nn.GRU` across 4 configs, initial-state handling, gate-semantics sanity. |

## Run it

```bash
python3 gru_scratch.py
python3 -m pytest test_gru.py -v -p no:anyio
```

## What to notice

- Matches `nn.GRU` to **~1e-5** when weights are shared.
- PyTorch places the reset gate *inside* the tanh (`r ⊙ (h W_hn + b_hn)`), not outside (`r ⊙ h) W_hn`) like some textbook versions. Different conventions, slightly different gradients; we follow PyTorch so weights copy cleanly.
- The encoder-decoder *framework* (encode the source sequence into a fixed vector, decode target from it) is language-agnostic. Bahdanau-style attention was bolted on a year later to fix the "fixed vector is a bottleneck" problem — which became the seed of the Transformer.

## References

- Cho et al. — [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078)
- Chung et al. — [Empirical Evaluation of Gated Recurrent Neural Networks](https://arxiv.org/abs/1412.3555) (the paper that convinced everyone GRU ≈ LSTM in practice)
