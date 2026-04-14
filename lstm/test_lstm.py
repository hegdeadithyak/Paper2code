"""
Tests for the scratch LSTM:
  - forward matches torch.nn.LSTM (same weights)
  - manual BPTT matches autograd
  - manual BPTT matches numerical finite-difference gradients
  - shapes and edge cases
"""

import pytest
import torch

from lstm_scratch import LSTMScratch
from lstm_library import LSTMLibrary


# ---------- fixtures ----------

@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(123)


def make_pair(I, H):
    """Library + scratch sharing the exact same weights."""
    lib = LSTMLibrary(I, H)
    sc = LSTMScratch(I, H)
    sc.load_from_torch_lstm(lib.lstm)
    return lib, sc


# ---------- forward ----------

@pytest.mark.parametrize("B,T,I,H", [
    (1, 1, 4, 4),      # degenerate single-step
    (3, 7, 5, 8),
    (8, 20, 16, 16),
    (2, 50, 1, 3),     # long sequence, tiny features
])
def test_forward_matches_library(B, T, I, H):
    lib, sc = make_pair(I, H)
    x = torch.randn(B, T, I)

    with torch.no_grad():
        lib_out, (lib_hN, lib_cN) = lib.forward(x)
    sc_out, (sc_hN, sc_cN) = sc.forward(x)

    assert sc_out.shape == (B, T, H)
    assert sc_hN.shape == (B, H)
    assert sc_cN.shape == (B, H)
    assert torch.allclose(sc_out, lib_out, atol=1e-5, rtol=1e-4)
    assert torch.allclose(sc_hN, lib_hN, atol=1e-5, rtol=1e-4)
    assert torch.allclose(sc_cN, lib_cN, atol=1e-5, rtol=1e-4)


def test_forward_with_initial_states():
    B, T, I, H = 4, 5, 6, 7
    lib, sc = make_pair(I, H)
    x = torch.randn(B, T, I)
    h0 = torch.randn(B, H)
    c0 = torch.randn(B, H)

    with torch.no_grad():
        lib_out, (lib_hN, lib_cN) = lib.forward(x, h0.unsqueeze(0), c0.unsqueeze(0))
    sc_out, (sc_hN, sc_cN) = sc.forward(x, h0, c0)

    assert torch.allclose(sc_out, lib_out, atol=1e-5, rtol=1e-4)
    assert torch.allclose(sc_hN, lib_hN, atol=1e-5, rtol=1e-4)
    assert torch.allclose(sc_cN, lib_cN, atol=1e-5, rtol=1e-4)


def test_forward_zero_input_gives_stable_state():
    B, T, I, H = 2, 10, 4, 4
    _, sc = make_pair(I, H)
    x = torch.zeros(B, T, I)
    out, (hN, cN) = sc.forward(x)
    assert torch.isfinite(out).all()
    assert torch.isfinite(hN).all()
    assert torch.isfinite(cN).all()


# ---------- backward vs autograd ----------

@pytest.mark.parametrize("B,T,I,H", [
    (2, 4, 3, 5),
    (5, 15, 8, 12),
])
def test_backward_matches_autograd(B, T, I, H):
    lib, sc = make_pair(I, H)

    x_np = torch.randn(B, T, I)
    d_out = torch.randn(B, T, H)

    # library: autograd
    x_lib = x_np.clone().requires_grad_(True)
    lib.lstm.zero_grad()
    lib_out, _ = lib.forward(x_lib)
    lib_out.backward(d_out)

    # scratch: manual BPTT
    sc.forward(x_np)
    g = sc.backward(d_out)

    assert torch.allclose(g["d_weight_ih"], lib.lstm.weight_ih_l0.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(g["d_weight_hh"], lib.lstm.weight_hh_l0.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(g["d_bias_ih"],   lib.lstm.bias_ih_l0.grad,   atol=1e-4, rtol=1e-3)
    assert torch.allclose(g["d_bias_hh"],   lib.lstm.bias_hh_l0.grad,   atol=1e-4, rtol=1e-3)
    assert torch.allclose(g["d_x"],         x_lib.grad,                 atol=1e-5, rtol=1e-4)


def test_backward_with_final_state_grads():
    """h_N and c_N grads should propagate correctly too."""
    B, T, I, H = 3, 6, 4, 5
    lib, sc = make_pair(I, H)

    x_np = torch.randn(B, T, I)
    d_hN = torch.randn(B, H)
    d_cN = torch.randn(B, H)

    # library
    x_lib = x_np.clone().requires_grad_(True)
    lib.lstm.zero_grad()
    out, (hN, cN) = lib.forward(x_lib)
    loss = (hN * d_hN).sum() + (cN * d_cN).sum()
    loss.backward()

    # scratch (d_out is zero, only final state grads flow)
    sc.forward(x_np)
    g = sc.backward(torch.zeros(B, T, H), d_hN=d_hN, d_cN=d_cN)

    assert torch.allclose(g["d_weight_ih"], lib.lstm.weight_ih_l0.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(g["d_x"],         x_lib.grad,                 atol=1e-5, rtol=1e-4)


# ---------- backward vs numerical (finite-difference) ----------

def test_backward_vs_finite_difference():
    """Gold-standard gradient check: compare analytical grads to numerical."""
    B, T, I, H = 2, 3, 3, 4
    _, sc = make_pair(I, H)
    # use float64 for a clean finite-difference signal
    sc.weight_ih = sc.weight_ih.double()
    sc.weight_hh = sc.weight_hh.double()
    sc.bias_ih = sc.bias_ih.double()
    sc.bias_hh = sc.bias_hh.double()

    x = torch.randn(B, T, I, dtype=torch.float64)
    d_out = torch.randn(B, T, H, dtype=torch.float64)

    # analytical
    sc.forward(x)
    g = sc.backward(d_out)

    # numerical grad w.r.t. weight_ih
    eps = 1e-6
    W = sc.weight_ih
    num_grad = torch.zeros_like(W)
    it = [(i, j) for i in range(W.shape[0]) for j in range(W.shape[1])]
    # subsample to keep the test fast
    sampled = it[::7]
    for i, j in sampled:
        orig = W[i, j].item()
        W[i, j] = orig + eps
        out_p, _ = sc.forward(x)
        lp = (out_p * d_out).sum().item()
        W[i, j] = orig - eps
        out_m, _ = sc.forward(x)
        lm = (out_m * d_out).sum().item()
        W[i, j] = orig
        num_grad[i, j] = (lp - lm) / (2 * eps)

    for i, j in sampled:
        assert abs(num_grad[i, j].item() - g["d_weight_ih"][i, j].item()) < 1e-5, \
            f"weight_ih[{i},{j}]: analytical {g['d_weight_ih'][i, j].item()} "\
            f"vs numerical {num_grad[i, j].item()}"


# ---------- weight loading ----------

def test_load_from_torch_lstm_copies_cleanly():
    lib, sc = make_pair(6, 8)
    assert torch.equal(sc.weight_ih, lib.lstm.weight_ih_l0.detach())
    assert torch.equal(sc.weight_hh, lib.lstm.weight_hh_l0.detach())
    assert torch.equal(sc.bias_ih, lib.lstm.bias_ih_l0.detach())
    assert torch.equal(sc.bias_hh, lib.lstm.bias_hh_l0.detach())


# ---------- shape sanity ----------

def test_param_shapes():
    I, H = 7, 11
    sc = LSTMScratch(I, H)
    assert sc.weight_ih.shape == (4 * H, I)
    assert sc.weight_hh.shape == (4 * H, H)
    assert sc.bias_ih.shape == (4 * H,)
    assert sc.bias_hh.shape == (4 * H,)
