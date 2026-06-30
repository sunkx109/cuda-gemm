"""Minimal usage example: call the custom CUDA GEMM operators from Python.

Performance measurements live in benchmark/bench_gemm.py.

Run:  python examples/example_gemm.py
"""

import torch

import cuda_gemm  # noqa: F401  (importing registers torch.ops.cuda_gemm.*)


def main():
    assert torch.cuda.is_available(), "A CUDA GPU is required to run this example."

    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")

    c_naive = torch.ops.cuda_gemm.gemm_naive(a, b)
    c_tiled = torch.ops.cuda_gemm.gemm_tiled(a, b)
    ref = torch.matmul(a, b)

    print("gemm_naive allclose:", torch.allclose(c_naive, ref, rtol=1e-4, atol=1e-4))
    print("gemm_tiled allclose:", torch.allclose(c_tiled, ref, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    main()
