"""
Tests: scratch Adam must produce the *exact same trajectory* as torch.optim.Adam
given the same seed, same init, same gradients.
"""

import pytest
import torch

from adam_scratch import AdamScratch
from adam_library import make_library_adam


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _trajectory(optimizer_factory, loss_fn, steps=50, n=5):
    x = torch.randn(n, requires_grad=True)
    opt = optimizer_factory([x])
    traj = [x.detach().clone()]
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(x)
        loss.backward()
        opt.step()
        traj.append(x.detach().clone())
        losses.append(loss.item())
    return torch.stack(traj), losses


def test_matches_torch_adam_on_quadratic():
    loss = lambda x: (x - 3).pow(2).sum()
    t_sc, _ = _trajectory(lambda p: AdamScratch(p, lr=0.1), loss)
    torch.manual_seed(0)
    t_lib, _ = _trajectory(lambda p: make_library_adam(p, lr=0.1), loss)
    assert torch.allclose(t_sc, t_lib, atol=1e-6, rtol=1e-5)


def test_matches_torch_adam_on_nonconvex():
    # sin(x) + x^2/10 — bumpy but bounded.
    loss = lambda x: (torch.sin(x * 3) + x.pow(2) / 10).sum()
    t_sc, _ = _trajectory(lambda p: AdamScratch(p, lr=0.05), loss, steps=100)
    torch.manual_seed(0)
    t_lib, _ = _trajectory(lambda p: make_library_adam(p, lr=0.05), loss, steps=100)
    assert torch.allclose(t_sc, t_lib, atol=1e-5, rtol=1e-4)


def test_converges_to_minimum():
    x = torch.tensor([0.0], requires_grad=True)
    opt = AdamScratch([x], lr=0.1)
    for _ in range(500):
        opt.zero_grad()
        ((x - 3).pow(2)).backward()
        opt.step()
    assert abs(x.item() - 3.0) < 1e-3


def test_bias_correction_active_early():
    """First step's update magnitude must be reasonable — not 1/(1-β2) tiny."""
    x = torch.tensor([1.0], requires_grad=True)
    opt = AdamScratch([x], lr=0.1)
    x_before = x.item()
    opt.zero_grad()
    (x * x).backward()   # grad = 2
    opt.step()
    # With bias correction, first step ≈ lr in magnitude. Without, it'd be ~lr/sqrt(1000).
    assert abs(x.item() - x_before) > 0.05


def test_custom_betas():
    loss = lambda x: (x - 1).pow(2).sum()
    t_sc, _ = _trajectory(lambda p: AdamScratch(p, lr=0.1, betas=(0.5, 0.9)), loss)
    torch.manual_seed(0)
    t_lib, _ = _trajectory(lambda p: make_library_adam(p, lr=0.1, betas=(0.5, 0.9)), loss)
    assert torch.allclose(t_sc, t_lib, atol=1e-6, rtol=1e-5)


def test_multi_param_groups():
    a = torch.randn(3, requires_grad=True)
    b = torch.randn(4, requires_grad=True)
    opt = AdamScratch([a, b], lr=0.1)
    a_norm0, b_norm0 = a.norm().item(), b.norm().item()
    for _ in range(200):
        opt.zero_grad()
        (a.pow(2).sum() + b.pow(2).sum()).backward()
        opt.step()
    assert a.norm().item() < a_norm0 * 0.1
    assert b.norm().item() < b_norm0 * 0.1
