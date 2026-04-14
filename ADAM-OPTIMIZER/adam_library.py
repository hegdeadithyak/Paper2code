"""
Adam via torch.optim.Adam — same math, C-speed, battle-tested.
The scratch version's update rule IS what this one does internally.
"""

import torch


def make_library_adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
    return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps)


if __name__ == "__main__":
    x = torch.tensor([0.0], requires_grad=True)
    opt = make_library_adam([x], lr=0.1)
    for i in range(200):
        opt.zero_grad()
        loss = (x - 3).pow(2).sum()
        loss.backward()
        opt.step()
    print(f"converged to x = {x.item():.6f}  (target = 3.0)")
