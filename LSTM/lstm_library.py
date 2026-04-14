"""
LSTM via torch.nn.LSTM — the canonical library path.

Forward is a one-liner. Backward is *free*: torch.autograd records the op graph
during forward and differentiates it automatically when you call .backward().
No manual BPTT. No gate-by-gate derivative math. That's the whole pitch of a
deep learning framework — diff `lstm_scratch.py`'s backward() against this.
"""

import torch
import torch.nn as nn


class LSTMLibrary:
    def __init__(self, input_size: int, hidden_size: int):
        self.I = input_size
        self.H = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bias=True,
        )

    def forward(self, x: torch.Tensor, h0=None, c0=None):
        B = x.size(0)
        if h0 is None:
            h0 = torch.zeros(1, B, self.H, dtype=x.dtype)
        if c0 is None:
            c0 = torch.zeros(1, B, self.H, dtype=x.dtype)
        out, (hN, cN) = self.lstm(x, (h0, c0))
        return out, (hN.squeeze(0), cN.squeeze(0))


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, I, H = 4, 6, 8, 5
    x = torch.randn(B, T, I, requires_grad=True)

    lstm = LSTMLibrary(I, H)
    out, (hN, cN) = lstm.forward(x)
    out.sum().backward()
    print(f"output: {tuple(out.shape)}   h_N: {tuple(hN.shape)}")
    print(f"d_x shape: {tuple(x.grad.shape)}   "
          f"d_weight_ih shape: {tuple(lstm.lstm.weight_ih_l0.grad.shape)}")
