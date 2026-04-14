<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=32&duration=3000&pause=800&color=6EE7B7&center=true&vCenter=true&width=720&lines=paper2code;Reading+papers+%3E+running+pip+install;papers%2C+rebuilt+from+first+principles" alt="paper2code" />

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

| Paper | From scratch | With library | Tests | Folder |
|------|:---:|:---:|:---:|---|
| Attention Is All You Need (Vaswani 2017) | 🟢 | 🟢 | 10 | [`attention/`](./attention) |
| An Image is Worth 16x16 Words — ViT (Dosovitskiy 2020) | 🟢 | 🟢 | 7 | [`vit/`](./vit) |
| Adam: A Method for Stochastic Optimization (Kingma 2014) | 🟢 | 🟢 | 6 | [`adam/`](./adam) |
| Long Short-Term Memory (Hochreiter 1997) | 🟢 | 🟢 | 12 | [`lstm/`](./lstm) |
| RNN Encoder–Decoder / GRU (Cho 2014) | 🟢 | 🟢 | 8 | [`gru/`](./gru) |
| Faster R-CNN (Ren 2015) — key atoms | 🟡 | 🟢 | 14 | [`faster-rcnn/`](./faster-rcnn) |
| SSD: Single Shot MultiBox Detector (Liu 2016) — key atoms | 🟡 | 🟢 | 11 | [`ssd/`](./ssd) |
| Bits & pieces from scratch | 🟢 | — | — | [`things-from-scratch/`](./things-from-scratch) |

<sub>🟢 done · 🟡 partial (detection papers — RPN/default-boxes/NMS/HNM from scratch; full pipeline via torchvision) · 🔴 todo</sub>

<sub>**Total: 68 tests, ~1.5s to run the whole repo suite.**</sub>

## 🚀 Get started

```bash
git clone https://github.com/hegdeadithyak/PaperReplica.git
cd PaperReplica
pip install -r requirements.txt

# run every test across every paper
python3 -m pytest -v
```

Each paper folder is self-contained: `{name}_scratch.py` + `{name}_library.py` + `test_{name}.py` + a README with the math, a diagram, and first-principles explanation. `cd` in and read the README.

## 🧱 The pattern

Every paper lives in a folder with the same four files:

```
<paper-dir>/
  <name>_scratch.py    # raw PyTorch tensor ops — no nn.Module, no autograd for RNNs
  <name>_library.py    # torch.nn / torchvision / torch.optim equivalent
  test_<name>.py       # shared-weight parity tests — scratch == library to ~1e-5
  README.md            # math, first-principles walkthrough, diagram from Wikimedia
```

Scratch impls expose their weights in the **same layout** as the library module, so tests can copy weights across with `load_from_torch_*()` and verify bit-identical outputs. No hand-waving, no "similar behavior" — numerical equivalence or the test fails.

## 📖 References

- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Dosovitskiy et al. — [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Kingma & Ba — [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- Hochreiter & Schmidhuber — [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
- Cho et al. — [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078)
- Ren et al. — [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- Liu et al. — [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

## 🪪 License

MIT — see [LICENSE](LICENSE).

<div align="center">
<br/>
<sub>built for the people who'd rather read the paper than the docs</sub>
</div>
