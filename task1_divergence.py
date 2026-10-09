"""
Task 1: Warp Divergence Microbenchmark
Run: python task1_divergence.py
"""
import time
import numpy as np
from numba import cuda

N = 2 ** 20
ITERS = 1000
THREADS = 256
TRIALS = 10


@cuda.jit
def kernel_a_uniform(y, n):
    """Kernel A: every thread executes the identical arithmetic."""
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        for _ in range(ITERS):
            v = v * 1.0001 + 0.0001
        y[idx] = v


@cuda.jit
def kernel_b_interleaved(y, n):
    """Kernel B: full divergence, adjacent threads take different paths."""
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        if idx % 2 == 0:
            for _ in range(ITERS):
                v = v * 1.0001 + 0.0001
        else:
            for _ in range(ITERS):
                v = (v - 0.0001) / 1.0001
        y[idx] = v


@cuda.jit
def kernel_c_warp_aligned(y, n):
    """Kernel C: whole warps take the same path, no intra-warp divergence."""
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for _ in range(ITERS):
                v = v * 1.0001 + 0.0001
        else:
            for _ in range(ITERS):
                v = (v - 0.0001) / 1.0001
        y[idx] = v


def benchmark(kernel, h_data):
    """Return average kernel-only time (ms) over TRIALS launches."""
    blocks = (N + THREADS - 1) // THREADS
    d_y = cuda.to_device(h_data)  # transfer is excluded from timing

    # warm-up launch (JIT compilation + cache warm)
    kernel[blocks, THREADS](d_y, N)
    cuda.synchronize()

    times = []
    for _ in range(TRIALS):
        cuda.synchronize()
        t0 = time.perf_counter()
        kernel[blocks, THREADS](d_y, N)
        cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return sum(times) / len(times)


def main():
    dev = cuda.get_current_device()
    print(f"GPU: {dev.name}, compute capability: {dev.compute_capability}")

    h_data = np.ones(N, dtype=np.float32)

    results = {}
    for name, kern in [
        ("Kernel A (Uniform)", kernel_a_uniform),
        ("Kernel B (Interleaved divergence)", kernel_b_interleaved),
        ("Kernel C (Warp-aligned)", kernel_c_warp_aligned),
    ]:
        results[name] = benchmark(kern, h_data)

    base = results["Kernel A (Uniform)"]
    print("\n| Kernel | Avg time (ms) | Slowdown vs A |")
    print("|---|---|---|")
    for name, t in results.items():
        print(f"| {name} | {t:.3f} | {t / base:.2f}x |")


if __name__ == "__main__":
    main()
