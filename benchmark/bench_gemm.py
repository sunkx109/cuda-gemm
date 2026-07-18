"""Benchmark the custom CUDA GEMM kernels vs torch.matmul (latency + TFLOPS).

Run:
    python benchmark/bench_gemm.py
    python benchmark/bench_gemm.py --sizes 512 1024 2048 4096 --iters 50
"""

import argparse

import torch

import cuda_gemm  # noqa: F401  (importing registers torch.ops.cuda_gemm.*)

# name -> callable(A, B) -> C
KERNELS = {
    "torch.matmul": torch.matmul,
    "gemm_naive": torch.ops.cuda_gemm.gemm_naive,
    "gemm_gmem_coalesce": torch.ops.cuda_gemm.gemm_gmem_coalesce,
    "gemm_smem": torch.ops.cuda_gemm.gemm_smem,
    "gemm_blocktiling_1d": torch.ops.cuda_gemm.gemm_blocktiling_1d,
    "gemm_blocktiling_2d": torch.ops.cuda_gemm.gemm_blocktiling_2d,
    "gemm_warptiling": torch.ops.cuda_gemm.gemm_warptiling,
    "gemm_cutlass": torch.ops.cuda_gemm.gemm_cutlass,
}


def bench(fn, a, b, warmup, iters):
    for _ in range(warmup):
        fn(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # milliseconds


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA GEMM kernels.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="square sizes M=N=K (default: 512 1024 2048 4096)",
    )
    parser.add_argument("--iters", type=int, default=50, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "A CUDA GPU is required."
    print(f"GPU: {torch.cuda.get_device_name(0)}   dtype: float32")

    col_w = 22
    ai_label = "AI(FLOPs/Bytes)"
    header = f"{'size (M=N=K)':<14} {ai_label:>13}" + "".join(
        f"{name:>{col_w}}" for name in KERNELS
    )
    print(header)
    print("-" * len(header))

    for s in args.sizes:
        a = torch.randn(s, s, device="cuda", dtype=torch.float32)
        b = torch.randn(s, s, device="cuda", dtype=torch.float32)
        flops = 2 * s**3

        # Arithmetic intensity = FLOPs / Bytes transferred
        # Read A (S² elem) + Read B (S² elem) + Write C (S² elem) = 3·S² elements
        # each fp32 element is 4 bytes → 12·S² bytes
        bytes_transferred = 12 * s**2
        ai = flops / bytes_transferred

        cells = []
        for fn in KERNELS.values():
            # Light correctness guard so we never benchmark a broken kernel.
            assert torch.allclose(fn(a, b), torch.matmul(a, b), rtol=1e-3, atol=1e-3)
            ms = bench(fn, a, b, args.warmup, args.iters)
            tflops = flops / (ms * 1e-3) / 1e12
            cells.append(f"{ms * 1e3:7.1f}us {tflops:7.1f}TF")

        print(f"{s:<14} {ai:13.1f}" + "".join(f"{c:>{col_w}}" for c in cells))


if __name__ == "__main__":
    main()
