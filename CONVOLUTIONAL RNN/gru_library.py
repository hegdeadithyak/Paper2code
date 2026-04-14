"""
GRU via torch.nn.GRU. Same math, fused kernels.
"""

import torch
import torch.nn as nn


class GRULibrary:
    def __init__(self, input_size: int, hidden_size: int):
        self.I = input_size
        self.H = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True, bias=True)

    def forward(self, x: torch.Tensor, h0=None):
        B = x.size(0)
        if h0 is None:
            h0 = torch.zeros(1, B, self.H, dtype=x.dtype)
        out, hN = self.gru(x, h0)
        return out, hN.squeeze(0)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, I, H = 3, 7, 5, 8
    x = torch.randn(B, T, I)
    gru = GRULibrary(I, H)
    out, hN = gru.forward(x)
    print(f"out: {tuple(out.shape)}   h_N: {tuple(hN.shape)}")
