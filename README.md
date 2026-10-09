# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** <Your Student ID>
**Allocated GPU Node:** <e.g., Tesla T4>
**CUDA Compute Capability:** <e.g., 7.5>
**Official Verification Token:** <PASTE_TOKEN_HERE>

## Task 1: Warp Divergence Benchmark

N = 2^20 float32 elements, 1,000 iterations per element, 256 threads per block.
Kernel-only time, averaged over 10 trials after a warm-up launch.

| Kernel | Avg time (ms) | Slowdown vs A |
|---|---|---|
| A (Uniform path) | <fill> | 1.00x |
| B (Interleaved, idx % 2) | <fill> | <fill> |
| C (Warp-aligned, warp_id % 2) | <fill> | <fill> |

**Analysis:** In Kernel B, adjacent threads in each warp take different branches, so the
hardware executes both paths one after another (serialization) and the warp pays for both.
In Kernel C every warp takes a single path (branching on `idx // 32`), so there is no
intra-warp divergence and the time is close to Kernel A.

## Task 2: 1D Stencil

N = 100,007 (odd, non-power-of-two). Boundary clamping replicates the edge values.

Result: `TASK 2 PASSED: MAX DELTA = <fill>`

## Task 3: Grid-Stride Loop

N = 2^24 = 16,777,216 elements, launched with 64 blocks x 256 threads = 16,384 threads.
Each thread processes elements `start, start + stride, ...` where `stride = cuda.gridsize(1)`.
All elements were verified to equal `factor`.

## Task 4: Sobel-X 2D

2048 x 2048 float32 image, 16 x 16 thread blocks, grid size computed with ceiling division.
Border pixels are set to 0.0; interior pixels use the Sobel-X kernel.

## How to run

```
pip install numba numpy
python task1_divergence.py
python task2_stencil_1d.py
python task3_grid_stride.py
python task4_sobel_2d.py
python verify_submission.py
```
