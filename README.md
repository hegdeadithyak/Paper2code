<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=32&duration=3000&pause=800&color=6EE7B7&center=true&vCenter=true&width=720&lines=paper2code;Reading+papers+%3E+running+pip+install;AI+papers%2C+rebuilt+from+first+principles" alt="paper2code" />

<br/>

<p>
  <img src="https://github.com/hegdeadithyak/PaperReplica/actions/workflows/tests.yml/badge.svg" />
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-22c55e?style=for-the-badge" />
  <img src="https://img.shields.io/badge/status-actively%20replicating-ff69b4?style=for-the-badge" />
</p>

<p>
  <img src="https://img.shields.io/github/stars/hegdeadithyak/PaperReplica?style=flat-square&color=facc15" />
  <img src="https://img.shields.io/github/forks/hegdeadithyak/PaperReplica?style=flat-square&color=60a5fa" />
  <img src="https://img.shields.io/github/last-commit/hegdeadithyak/PaperReplica?style=flat-square&color=a78bfa" />
  <img src="https://img.shields.io/github/repo-size/hegdeadithyak/PaperReplica?style=flat-square&color=f472b6" />
</p>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" />

</div>

## What this is

**paper2code** is a collection of AI/ML research papers rebuilt in Python — stripped of the abstractions that hide what's actually happening.

Every paper gets replicated **twice** (or close to it):

- **🧱 From scratch** — raw NumPy / pure PyTorch tensors. No `nn.Module` magic, no `torch.optim`, no black boxes. You read the math, you write the math.
- **📦 With libraries** — the canonical high-level implementation, so you can diff the two and *see* exactly what the library is doing for you.

The goal isn't benchmark-chasing. It's **understanding** — by the time you've written backprop by hand once, `loss.backward()` stops feeling like magic.

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" />
</div>

## 📚 Paper hitlist

| Paper | From scratch | With library | Folder |
|------|:---:|:---:|---|
| Attention Is All You Need | 🟢 | 🟢 | [`ATTENTION IS ALL YOU NEED/`](./ATTENTION%20IS%20ALL%20YOU%20NEED) |
| An Image is Worth 16x16 Words (ViT) | 🟡 | 🟢 | [`VisionTransformer/`](./VisionTransformer) |
| ADAM: A Method for Stochastic Optimization | 🟢 | 🟢 | [`ADAM-OPTIMIZER/`](./ADAM-OPTIMIZER) |
| Long Short-Term Memory | 🟢 | 🟡 | [`LSTM/`](./LSTM) |
| RNN Encoder–Decoder for SMT | 🟡 | 🟡 | [`CONVOLUTIONAL RNN/`](./CONVOLUTIONAL%20RNN) |
| Faster R-CNN | 🔴 | 🟡 | [`FASTER-RCNN/`](./FASTER-RCNN) |
| SSD-ResNet Object Detection | 🔴 | 🟡 | [`SSD-RESNET/`](./SSD-RESNET) |
| Bits & pieces from scratch | 🟢 | — | [`THINGS_FROM_SCRATCH/`](./THINGS_FROM_SCRATCH) |

<sub>🟢 done · 🟡 in progress · 🔴 todo</sub>

## 🚀 Get started

```bash
git clone https://github.com/hegdeadithyak/PaperReplica.git
cd PaperReplica
pip install -r requirements.txt
```

Then `cd` into any paper folder and read its README — each one is self-contained.

## 📖 References

- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- Dosovitskiy et al. — [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Kingma & Ba — [ADAM: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- Cho et al. — [RNN Encoder–Decoder for SMT](https://arxiv.org/pdf/1406.1078)
- [Object detection based on SSD-ResNet](https://ieeexplore.ieee.org/document/9073753)

## 🪪 License

MIT — see [LICENSE](LICENSE).

<div align="center">
<br/>
<sub>built for the people who'd rather read the paper than the docs</sub>
</div>
