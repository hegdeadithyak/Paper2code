"""
Load identical weights into both implementations, run same input,
compare forward outputs AND backward gradients, and time each direction.
"""

import time
import torch

from lstm_scratch import LSTMScratch
from lstm_library import LSTMLibrary


def bench(fn, warmup=2, runs=10):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) / runs


def main():
    torch.manual_seed(42)

    B, T, I, H = 16, 50, 32, 64
    print(f"config: batch={B}  seq_len={T}  input_size={I}  hidden_size={H}\n")

    # --- shared weights ---
    lib = LSTMLibrary(I, H)
    scratch = LSTMScratch(I, H)
    scratch.load_from_torch_lstm(lib.lstm)

    # --- inputs ---
    x_np = torch.randn(B, T, I)

    # ========== FORWARD comparison ==========
    x_lib = x_np.clone().requires_grad_(True)
    lib_out, (lib_hN, lib_cN) = lib.forward(x_lib)

    x_sc = x_np.clone()
    sc_out, (sc_hN, sc_cN) = scratch.forward(x_sc)

    # ========== BACKWARD ==========
    d_out = torch.randn(B, T, H)       # arbitrary upstream grad
    lib_out.backward(d_out)

    sc_grads = scratch.backward(d_out)

    # ========== NUMERICAL DIFFS ==========
    def mx(a, b): return (a - b).abs().max().item()

    fwd_diff = mx(lib_out.detach(), sc_out)
    dwih_diff = mx(lib.lstm.weight_ih_l0.grad, sc_grads["d_weight_ih"])
    dwhh_diff = mx(lib.lstm.weight_hh_l0.grad, sc_grads["d_weight_hh"])
    dbih_diff = mx(lib.lstm.bias_ih_l0.grad, sc_grads["d_bias_ih"])
    dbhh_diff = mx(lib.lstm.bias_hh_l0.grad, sc_grads["d_bias_hh"])
    dx_diff = mx(x_lib.grad, sc_grads["d_x"])

    # ========== TIMING ==========
    def lib_fwd_bwd():
        xx = x_np.clone().requires_grad_(True)
        lib.lstm.zero_grad()
        o, _ = lib.forward(xx)
        o.backward(d_out)

    def sc_fwd_bwd():
        o, _ = scratch.forward(x_np)
        scratch.backward(d_out)

    def lib_fwd_only():
        with torch.no_grad():
            lib.forward(x_np)

    def sc_fwd_only():
        scratch.forward(x_np)

    lib_fwd_t = bench(lib_fwd_only)
    sc_fwd_t = bench(sc_fwd_only)
    lib_fb_t = bench(lib_fwd_bwd)
    sc_fb_t = bench(sc_fwd_bwd)

    # ========== REPORT ==========
    print("=" * 62)
    print(f"{'':<26}{'scratch':>16}{'library':>18}")
    print("-" * 62)
    print(f"{'forward (ms)':<26}{sc_fwd_t*1000:>16.3f}{lib_fwd_t*1000:>18.3f}")
    print(f"{'forward+backward (ms)':<26}{sc_fb_t*1000:>16.3f}{lib_fb_t*1000:>18.3f}")
    print(f"{'fwd speedup (lib/sc)':<26}{'':>16}{sc_fwd_t/lib_fwd_t:>17.2f}x")
    print(f"{'fwd+bwd speedup':<26}{'':>16}{sc_fb_t/lib_fb_t:>17.2f}x")
    print("=" * 62)

    print("\nforward agreement:")
    print(f"  max |Δ output|         = {fwd_diff:.3e}")
    print("\nbackward agreement (manual BPTT vs autograd, same weights/input/dOut):")
    print(f"  max |Δ d_weight_ih|    = {dwih_diff:.3e}")
    print(f"  max |Δ d_weight_hh|    = {dwhh_diff:.3e}")
    print(f"  max |Δ d_bias_ih|      = {dbih_diff:.3e}")
    print(f"  max |Δ d_bias_hh|      = {dbhh_diff:.3e}")
    print(f"  max |Δ d_x|            = {dx_diff:.3e}")

    worst = max(fwd_diff, dwih_diff, dwhh_diff, dbih_diff, dbhh_diff, dx_diff)
    print(f"\n  worst diff overall     = {worst:.3e}  "
          f"({'✅ match' if worst < 1e-4 else '❌ mismatch'})")


if __name__ == "__main__":
    main()
