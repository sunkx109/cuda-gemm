"""Single-kernel profiling harness for NVIDIA Nsight Compute (ncu).

Launches ONLY the selected custom kernel (after a short warmup), so ncu captures
a clean profile of that one kernel — no torch.matmul / cuBLAS or other-variant
noise. For latency/TFLOPS comparisons use bench_gemm.py instead.

Quick start (profile gemm_tiled at 4096x4096x4096):

    ncu --set full --target-processes all \\
        --kernel-name regex:matmul_tiled_kernel \\
        --launch-skip 3 --launch-count 1 \\
        -o profile_gemm_tiled \\
        python benchmark/profile_kernel.py --kernel gemm_tiled --size 4096

Notes:
    - `--kernel-name regex:matmul_<variant>_kernel` filters out the fill kernels
      that torch.randn launches, so only your GEMM kernel is collected.
    - `--launch-skip <warmup>` skips warmup launches; `--launch-count <repeats>`
      limits how many launches ncu profiles. With the defaults (warmup=3,
      repeats=1) ncu replays a single profiled launch for each metric set.
    - ncu may need elevated privileges for HW counters. If you get permission
      errors, run with sudo, or:
          echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
"""

import argparse

import torch

import cuda_gemm  # noqa: F401  (registers torch.ops.cuda_gemm.*)

# Python op name -> callable. Add new variants here as you register them.
KERNELS = {
    "gemm_naive": torch.ops.cuda_gemm.gemm_naive,
    "gemm_gmem_coalesce": torch.ops.cuda_gemm.gemm_gmem_coalesce,
    "gemm_smem": torch.ops.cuda_gemm.gemm_smem,
    "gemm_tiled": torch.ops.cuda_gemm.gemm_tiled,
}

# Python op name -> underlying CUDA device-kernel name (for ncu --kernel-name).
DEVICE_KERNEL = {
    "gemm_naive": "matmul_naive_kernel",
    "gemm_gmem_coalesce": "matmul_gmem_coalesce_kernel",
    "gemm_smem": "sgemm_shared_mem_kernel",
    "gemm_tiled": "matmul_tiled_kernel",
}


def main():
    parser = argparse.ArgumentParser(description="Profile a single CUDA GEMM kernel for ncu.")
    parser.add_argument(
        "--kernel", required=True, choices=list(KERNELS), help="kernel to launch in isolation"
    )
    parser.add_argument("--size", type=int, default=2048, help="square size M=N=K (default 2048)")
    parser.add_argument("--m", type=int, default=None, help="override M (defaults to --size)")
    parser.add_argument("--k", type=int, default=None, help="override K (defaults to --size)")
    parser.add_argument("--n", type=int, default=None, help="override N (defaults to --size)")
    parser.add_argument(
        "--warmup", type=int, default=3, help="warmup launches before the profiled ones"
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="profiled launches (keep 1 for a clean ncu profile)"
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "A CUDA GPU is required."
    torch.manual_seed(0)

    M = args.m or args.size
    K = args.k or args.size
    N = args.n or args.size

    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    fn = KERNELS[args.kernel]

    dev_kernel = DEVICE_KERNEL.get(args.kernel, args.kernel)
    print(
        f"[profile] kernel={args.kernel} ({dev_kernel})  "
        f"A=({M},{K}) B=({K},{N}) -> C=({M},{N})  warmup={args.warmup} repeats={args.repeats}"
    )
    print(
        "[profile] suggested ncu:\n"
        f"  ncu --set full --target-processes all "
        f"--kernel-name regex:{dev_kernel} "
        f"--launch-skip {args.warmup} --launch-count {args.repeats} "
        f"-o profile_{args.kernel} "
        f"python benchmark/profile_kernel.py --kernel {args.kernel} --size {args.size}"
    )

    for _ in range(args.warmup):
        fn(a, b)
    torch.cuda.synchronize()

    for _ in range(args.repeats):
        fn(a, b)
    torch.cuda.synchronize()

    print("[profile] done.")


if __name__ == "__main__":
    main()
