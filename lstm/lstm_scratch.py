"""
LSTM from scratch — pure torch tensor ops, no nn.Module, no nn.LSTM, no autograd.

Forward (Hochreiter & Schmidhuber, 1997; Gers et al., 2000):
    i_t = σ(x_t W_ii + b_ii + h_{t-1} W_hi + b_hi)
    f_t = σ(x_t W_if + b_if + h_{t-1} W_hf + b_hf)
    g_t = tanh(x_t W_ig + b_ig + h_{t-1} W_hg + b_hg)
    o_t = σ(x_t W_io + b_io + h_{t-1} W_ho + b_ho)
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)

Backward — manual BPTT. For each step, given incoming dh_t and dc_{t+1}:
    dc_t   = dh_t ⊙ o_t ⊙ (1 - tanh(c_t)²) + dc_{t+1}
    do     = dh_t ⊙ tanh(c_t)
    df     = dc_t ⊙ c_{t-1}
    di     = dc_t ⊙ g_t
    dg     = dc_t ⊙ i_t
    dc_{t-1} = dc_t ⊙ f_t
Pass each through its activation derivative to get pre-activation grads,
then accumulate parameter grads and propagate to x_t and h_{t-1}.

Weight layout matches torch.nn.LSTM: rows stacked as [i, f, g, o].
"""

import torch


class LSTMScratch:
    def __init__(self, input_size: int, hidden_size: int, dtype=torch.float32):
        self.I = input_size
        self.H = hidden_size
        H4 = 4 * hidden_size
        k = 1.0 / (hidden_size ** 0.5)
        self.weight_ih = (torch.rand(H4, input_size, dtype=dtype) * 2 - 1) * k
        self.weight_hh = (torch.rand(H4, hidden_size, dtype=dtype) * 2 - 1) * k
        self.bias_ih = (torch.rand(H4, dtype=dtype) * 2 - 1) * k
        self.bias_hh = (torch.rand(H4, dtype=dtype) * 2 - 1) * k
        self._cache = None

    def load_from_torch_lstm(self, lstm: torch.nn.LSTM):
        sd = lstm.state_dict()
        self.weight_ih = sd["weight_ih_l0"].detach().clone()
        self.weight_hh = sd["weight_hh_l0"].detach().clone()
        self.bias_ih = sd["bias_ih_l0"].detach().clone()
        self.bias_hh = sd["bias_hh_l0"].detach().clone()

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None, c0: torch.Tensor = None):
        B, T, I = x.shape
        H = self.H
        assert I == self.I

        h = torch.zeros(B, H, dtype=x.dtype) if h0 is None else h0.clone()
        c = torch.zeros(B, H, dtype=x.dtype) if c0 is None else c0.clone()

        outputs = torch.empty(B, T, H, dtype=x.dtype)

        # caches for BPTT
        xs = x
        h_prevs = torch.empty(T, B, H, dtype=x.dtype)
        c_prevs = torch.empty(T, B, H, dtype=x.dtype)
        i_s = torch.empty(T, B, H, dtype=x.dtype)
        f_s = torch.empty(T, B, H, dtype=x.dtype)
        g_s = torch.empty(T, B, H, dtype=x.dtype)
        o_s = torch.empty(T, B, H, dtype=x.dtype)
        c_s = torch.empty(T, B, H, dtype=x.dtype)

        Wih_T = self.weight_ih.t()
        Whh_T = self.weight_hh.t()
        b = self.bias_ih + self.bias_hh

        for t in range(T):
            xt = x[:, t, :]
            gates = xt @ Wih_T + h @ Whh_T + b
            i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
            i_a = torch.sigmoid(i_g)
            f_a = torch.sigmoid(f_g)
            g_a = torch.tanh(g_g)
            o_a = torch.sigmoid(o_g)

            h_prevs[t] = h
            c_prevs[t] = c
            c = f_a * c + i_a * g_a
            h = o_a * torch.tanh(c)

            i_s[t], f_s[t], g_s[t], o_s[t], c_s[t] = i_a, f_a, g_a, o_a, c
            outputs[:, t, :] = h

        self._cache = (xs, h_prevs, c_prevs, i_s, f_s, g_s, o_s, c_s, h0, c0)
        return outputs, (h, c)

    def backward(self, d_out: torch.Tensor, d_hN: torch.Tensor = None, d_cN: torch.Tensor = None):
        """
        Args:
            d_out: gradient w.r.t. outputs, shape [B, T, H]
            d_hN : gradient w.r.t. final hidden h_N, shape [B, H] (optional)
            d_cN : gradient w.r.t. final cell   c_N, shape [B, H] (optional)
        Returns:
            dict with d_weight_ih, d_weight_hh, d_bias_ih, d_bias_hh, d_x, d_h0, d_c0
        """
        xs, h_prevs, c_prevs, i_s, f_s, g_s, o_s, c_s, h0, c0 = self._cache
        B, T, H = d_out.shape
        I = self.I

        dW_ih = torch.zeros_like(self.weight_ih)
        dW_hh = torch.zeros_like(self.weight_hh)
        db = torch.zeros_like(self.bias_ih)   # b_ih + b_hh share grad
        dx = torch.zeros_like(xs)

        dh_next = torch.zeros(B, H, dtype=xs.dtype) if d_hN is None else d_hN.clone()
        dc_next = torch.zeros(B, H, dtype=xs.dtype) if d_cN is None else d_cN.clone()

        for t in reversed(range(T)):
            i_a, f_a, g_a, o_a = i_s[t], f_s[t], g_s[t], o_s[t]
            c_t, c_prev, h_prev = c_s[t], c_prevs[t], h_prevs[t]

            dh = d_out[:, t, :] + dh_next
            tanh_c = torch.tanh(c_t)
            dc = dh * o_a * (1 - tanh_c * tanh_c) + dc_next

            do = dh * tanh_c
            df = dc * c_prev
            di = dc * g_a
            dg = dc * i_a

            # through activations → pre-activation grads
            di_pre = di * i_a * (1 - i_a)
            df_pre = df * f_a * (1 - f_a)
            dg_pre = dg * (1 - g_a * g_a)
            do_pre = do * o_a * (1 - o_a)

            dgates = torch.cat([di_pre, df_pre, dg_pre, do_pre], dim=1)   # [B, 4H]

            xt = xs[:, t, :]
            dW_ih += dgates.t() @ xt
            dW_hh += dgates.t() @ h_prev
            db += dgates.sum(dim=0)

            dx[:, t, :] = dgates @ self.weight_ih
            dh_next = dgates @ self.weight_hh
            dc_next = dc * f_a

        return {
            "d_weight_ih": dW_ih,
            "d_weight_hh": dW_hh,
            "d_bias_ih": db,
            "d_bias_hh": db.clone(),
            "d_x": dx,
            "d_h0": dh_next,
            "d_c0": dc_next,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, I, H = 4, 6, 8, 5
    x = torch.randn(B, T, I)

    lstm = LSTMScratch(I, H)
    out, (hN, cN) = lstm.forward(x)
    grads = lstm.backward(torch.ones_like(out))
    print(f"output: {tuple(out.shape)}   h_N: {tuple(hN.shape)}")
    print(f"d_weight_ih: {tuple(grads['d_weight_ih'].shape)}   "
          f"d_x: {tuple(grads['d_x'].shape)}")
