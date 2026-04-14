"""
Scratch GRU must match torch.nn.GRU when weights are shared.
"""

import pytest
import torch

from gru_scratch import GRUScratch
from gru_library import GRULibrary


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _pair(I, H):
    lib = GRULibrary(I, H)
    sc = GRUScratch(I, H)
    sc.load_from_torch_gru(lib.gru)
    return lib, sc


@pytest.mark.parametrize("B,T,I,H", [
    (1, 1, 4, 4),
    (3, 7, 5, 8),
    (8, 30, 16, 16),
    (2, 50, 1, 3),
])
def test_forward_matches_library(B, T, I, H):
    lib, sc = _pair(I, H)
    x = torch.randn(B, T, I)
    with torch.no_grad():
        lib_out, lib_hN = lib.forward(x)
    sc_out, sc_hN = sc.forward(x)
    assert sc_out.shape == (B, T, H)
    assert torch.allclose(sc_out, lib_out, atol=1e-5, rtol=1e-4)
    assert torch.allclose(sc_hN, lib_hN, atol=1e-5, rtol=1e-4)


def test_with_initial_state():
    lib, sc = _pair(6, 8)
    B, T = 3, 5
    x = torch.randn(B, T, 6)
    h0 = torch.randn(B, 8)
    with torch.no_grad():
        lib_out, lib_hN = lib.forward(x, h0.unsqueeze(0))
    sc_out, sc_hN = sc.forward(x, h0)
    assert torch.allclose(sc_out, lib_out, atol=1e-5, rtol=1e-4)


def test_zero_input_stays_finite():
    _, sc = _pair(4, 4)
    out, _ = sc.forward(torch.zeros(2, 10, 4))
    assert torch.isfinite(out).all()


def test_param_shapes():
    sc = GRUScratch(7, 11)
    assert sc.weight_ih.shape == (3 * 11, 7)
    assert sc.weight_hh.shape == (3 * 11, 11)
    assert sc.bias_ih.shape == (3 * 11,)


def test_update_gate_bounds_behavior():
    """If z_t ≈ 1 everywhere (update gate fully closed), h barely changes.
    If z_t ≈ 0, h is replaced by the candidate every step."""
    H = 4
    # force z ≈ 1: huge positive bias on update-gate slot [H:2H]
    sc_hold = GRUScratch(3, H)
    sc_hold.bias_ih = torch.zeros(3 * H)
    sc_hold.bias_hh = torch.zeros(3 * H)
    sc_hold.bias_ih[H:2*H] = 100.0
    h0 = torch.randn(1, H)
    out, _ = sc_hold.forward(torch.randn(1, 5, 3), h0=h0)
    assert torch.allclose(out[:, -1], h0, atol=1e-3)  # nothing changed
