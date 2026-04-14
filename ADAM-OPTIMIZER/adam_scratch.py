"""
Adam from scratch — Kingma & Ba, 2014. ~20 lines of actual math.

Why Adam works:
  SGD takes the gradient and walks. If some parameter has huge gradients
  sometimes and tiny ones other times, SGD wobbles. RMSProp fixes that by
  dividing by the running RMS of past gradients — each param gets its own
  adaptive step size.

  Momentum fixes the OTHER problem — pure SGD oscillates across narrow
  valleys instead of rolling down them. A running mean of past gradients
  (momentum) smooths the trajectory.

  Adam = RMSProp + momentum + a bias-correction trick so the early steps
  aren't underestimated (m and v start at zero; without the correction
  the first few updates are tiny).

Update rule, for each parameter θ:
    g_t  = ∇θ L
    m_t  = β1·m_{t-1} + (1-β1)·g_t          # momentum  (1st moment)
    v_t  = β2·v_{t-1} + (1-β2)·g_t²         # RMS       (2nd moment)
    m̂_t = m_t / (1 - β1^t)                  # bias-corrected
    v̂_t = v_t / (1 - β2^t)
    θ   = θ - lr · m̂_t / (√v̂_t + ε)
"""

import torch


class AdamScratch:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m_hat = m / bc1
            v_hat = v / bc2
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


if __name__ == "__main__":
    # Minimize f(x) = (x - 3)^2  — global min at x=3.
    x = torch.tensor([0.0], requires_grad=True)
    opt = AdamScratch([x], lr=0.1)
    for i in range(200):
        opt.zero_grad()
        loss = (x - 3).pow(2).sum()
        loss.backward()
        opt.step()
    print(f"converged to x = {x.item():.6f}  (target = 3.0)")
