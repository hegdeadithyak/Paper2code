"""
GRU from scratch — Cho et al., 2014, *Learning Phrase Representations using
RNN Encoder-Decoder for Statistical Machine Translation*.

The paper introduced two things:
  1. The encoder-decoder framework that became seq2seq.
  2. The GRU — a simpler LSTM with two gates instead of four.

Gate equations (PyTorch's convention — note: PyTorch applies n's reset gate
slightly differently than some textbooks; we follow PyTorch so weights copy):
    r_t = σ(x_t W_ir + b_ir + h_{t-1} W_hr + b_hr)     reset gate
    z_t = σ(x_t W_iz + b_iz + h_{t-1} W_hz + b_hz)     update gate
    n_t = tanh(x_t W_in + b_in + r_t ⊙ (h_{t-1} W_hn + b_hn))   candidate
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}

Weight layout matches torch.nn.GRU: [3H, I] stacked as [r, z, n].
"""

import torch


class GRUScratch:
    def __init__(self, input_size: int, hidden_size: int, dtype=torch.float32):
        self.I = input_size
        self.H = hidden_size
        H3 = 3 * hidden_size
        k = 1.0 / (hidden_size ** 0.5)
        self.weight_ih = (torch.rand(H3, input_size, dtype=dtype) * 2 - 1) * k
        self.weight_hh = (torch.rand(H3, hidden_size, dtype=dtype) * 2 - 1) * k
        self.bias_ih = (torch.rand(H3, dtype=dtype) * 2 - 1) * k
        self.bias_hh = (torch.rand(H3, dtype=dtype) * 2 - 1) * k

    def load_from_torch_gru(self, gru: torch.nn.GRU):
        sd = gru.state_dict()
        self.weight_ih = sd["weight_ih_l0"].detach().clone()
        self.weight_hh = sd["weight_hh_l0"].detach().clone()
        self.bias_ih = sd["bias_ih_l0"].detach().clone()
        self.bias_hh = sd["bias_hh_l0"].detach().clone()

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        B, T, I = x.shape
        H = self.H
        h = torch.zeros(B, H, dtype=x.dtype) if h0 is None else h0.clone()
        outputs = torch.empty(B, T, H, dtype=x.dtype)

        Wih_T = self.weight_ih.t()   # [I, 3H]
        Whh_T = self.weight_hh.t()   # [H, 3H]

        for t in range(T):
            xt = x[:, t, :]
            gi = xt @ Wih_T + self.bias_ih       # [B, 3H]
            gh = h @ Whh_T + self.bias_hh        # [B, 3H]

            i_r, i_z, i_n = gi.chunk(3, dim=1)
            h_r, h_z, h_n = gh.chunk(3, dim=1)

            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)        # PyTorch's reset placement
            h = (1 - z) * n + z * h

            outputs[:, t, :] = h

        return outputs, h


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, I, H = 3, 7, 5, 8
    x = torch.randn(B, T, I)
    gru = GRUScratch(I, H)
    out, hN = gru.forward(x)
    print(f"out: {tuple(out.shape)}   h_N: {tuple(hN.shape)}")
