"""
Task 4: 2D spatial convolution - Sobel horizontal (Sobel-X) filter
Run: python task4_sobel_2d.py
"""
import numpy as np
from numba import cuda


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)
    if row < rows and col < cols:
        if row > 0 and row < rows - 1 and col > 0 and col < cols - 1:
            d_out[row, col] = (
                -1.0 * d_in[row - 1, col - 1] + 1.0 * d_in[row - 1, col + 1]
                - 2.0 * d_in[row, col - 1] + 2.0 * d_in[row, col + 1]
                - 1.0 * d_in[row + 1, col - 1] + 1.0 * d_in[row + 1, col + 1]
            )
        else:
            d_out[row, col] = 0.0


def run_sobel(h_img):
    h_img = np.ascontiguousarray(h_img, dtype=np.float32)
    rows, cols = h_img.shape

    threads_2d = (16, 16)
    blocks_2d = (
        (cols + threads_2d[0] - 1) // threads_2d[0],  # x -> columns
        (rows + threads_2d[1] - 1) // threads_2d[1],  # y -> rows
    )

    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array((rows, cols), dtype=np.float32)
    sobel_x_kernel[blocks_2d, threads_2d](d_in, d_out, rows, cols)
    cuda.synchronize()
    return d_out.copy_to_host()


def main():
    rng = np.random.default_rng(0)
    h_img = rng.random((2048, 2048), dtype=np.float32)
    out = run_sobel(h_img)

    # CPU reference for interior pixels
    ref = np.zeros_like(h_img)
    a = h_img
    ref[1:-1, 1:-1] = (
        -a[:-2, :-2] + a[:-2, 2:]
        - 2.0 * a[1:-1, :-2] + 2.0 * a[1:-1, 2:]
        - a[2:, :-2] + a[2:, 2:]
    )
    assert np.allclose(out, ref, atol=1e-4)
    print(f"TASK 4 PASSED: shape={out.shape}, MAX DELTA = {np.max(np.abs(out - ref))}")


if __name__ == "__main__":
    main()
