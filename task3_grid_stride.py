"""
Task 3: Arbitrary-size vector scaling via a grid-stride loop
Run: python task3_grid_stride.py
"""
import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 64  # 16,384 total threads, far fewer than N


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    N = h_arr.shape[0]
    d_arr = cuda.to_device(h_arr)
    grid_stride_scale_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
        d_arr, np.float32(factor), N
    )
    cuda.synchronize()
    return d_arr.copy_to_host()


def main():
    N = 2 ** 24  # 16,777,216
    factor = 4.25
    h_arr = np.ones(N, dtype=np.float32)

    total_threads = BLOCKS_PER_GRID * THREADS_PER_BLOCK
    print(f"Total hardware threads launched: {total_threads:,} (N = {N:,})")

    res = run_grid_stride(h_arr, factor)
    assert res.shape[0] == N
    assert np.all(res == np.float32(factor)), "Not all elements match factor"
    print(f"TASK 3 PASSED: all {N:,} elements == {factor}")


if __name__ == "__main__":
    main()
