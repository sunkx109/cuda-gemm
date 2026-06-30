# cuda-gemm

A repository of **progressively-optimized CUDA GEMM (matrix multiplication) kernels**, packaged as a PyTorch custom-operator extension.

The goal is to start from the most naive implementation and optimize it step by step — tiling, memory layout, vectorization, Tensor Cores — and eventually work toward [CUTLASS](https://github.com/NVIDIA/cutlass), getting as close as possible to the performance of `torch.matmul`. Along the way, this repo is a hands-on way to get a real feel for **kernel performance analysis, performance optimization, and GPU hardware architecture**.

Each optimization step lives in its own file under `csrc/kernels/`, is registered as a separate operator (`torch.ops.cuda_gemm.*`), and can be individually tested, benchmarked, and profiled against `torch.matmul`.

## Current kernels

| Op            | Strategy                       | Notes                          |
| ------------- | ------------------------------ | ------------------------------ |
| `gemm_naive`  | one thread per output element  | baseline, the starting point   |
| `gemm_tiled`  | 16×16 shared-memory tiling     | first optimization step        |

> **Roadmap:** vectorized loads (`float4`) → compute/data tiling & thread coarsening → warp-level tiling → Tensor Cores (`wmma` / `mma`) → CUTLASS. Add a new variant by following [Adding a new kernel](#adding-a-new-kernel).

## Requirements

- NVIDIA GPU + CUDA (validated on CUDA 12.9 / PyTorch 2.11)
- Build: `nvcc` and a GCC compatible with your PyTorch
- Runtime: PyTorch + GPU


## Quick start

```bash
pip install -r requirements.txt        # 1. install deps
python setup.py develop                # 2. build & install (re-run after editing kernels)
python examples/example_gemm.py        # 3. usage example
pytest -q tests/                       # 4. correctness tests
python benchmark/bench_gemm.py         # 5. benchmark vs torch.matmul
```

## Usage

```python
import torch
import cuda_gemm   # side-effect import: registers torch.ops.cuda_gemm.*

a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")

c1 = torch.ops.cuda_gemm.gemm_naive(a, b)
c2 = torch.ops.cuda_gemm.gemm_tiled(a, b)
assert torch.allclose(c2, torch.matmul(a, b), rtol=1e-4, atol=1e-4)
```

> `import cuda_gemm` is **required**: it loads the compiled `.so`, which triggers the `TORCH_LIBRARY` static registration that makes `torch.ops.cuda_gemm.*` available. It must come **after** `import torch` — the extension links torch's shared libraries.

## Adding a new kernel

1. Create `csrc/kernels/foo.cu` with a `__global__` kernel and a host launcher `torch::Tensor foo(torch::Tensor A, torch::Tensor B)` (`#include "ops.h"`).
2. Declare the launcher in `csrc/ops.h`.
3. Register it in `csrc/ops.cpp`:
   - `m.def("foo(Tensor A, Tensor B) -> Tensor")` inside `TORCH_LIBRARY(cuda_gemm, m)`,
   - `m.impl("foo", &foo)` inside `TORCH_LIBRARY_IMPL(cuda_gemm, CUDA, m)`.
4. Re-run `python setup.py develop` (no need to edit `sources` — globs pick it up), then call `torch.ops.cuda_gemm.foo(...)`.
5. Add it to the `OPS` list in `tests/test_gemm.py` for correctness across all shapes, and (optionally) to the `KERNELS` / `DEVICE_KERNEL` maps in the benchmark scripts.

## Benchmark

```bash
python benchmark/bench_gemm.py                                  # default 512/1024/2048
python benchmark/bench_gemm.py --sizes 512 1024 2048 4096 --iters 50
```

Reports latency (µs) and achieved TFLOPS for each kernel vs `torch.matmul`, with a light correctness guard.

## Single-kernel profiling (Nsight Compute / `ncu`)

`benchmark/profile_kernel.py` launches **only** the kernel you pick (no `torch.matmul` or other variants), so `ncu` captures a clean profile of that one kernel:

```bash
# the script prints a ready-to-copy ncu command:
python benchmark/profile_kernel.py --kernel gemm_tiled --size 4096

# collect with ncu (--kernel-name filters out torch.randn fill kernels):
ncu --set full --target-processes all \
    --kernel-name regex:matmul_tiled_kernel --launch-skip 3 --launch-count 1 \
    -o profile_gemm_tiled \
    python benchmark/profile_kernel.py --kernel gemm_tiled --size 4096
```

> `--kernel-name regex:matmul_<variant>_kernel` matches the device-kernel name in `csrc/kernels/*.cu`. If `ncu` reports a permission error: `echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid`, or run under `sudo`.

## Code style (pre-commit)

Commits are checked and formatted by pre-commit: **ruff** (Python), **clang-format** (C++/CUDA), plus whitespace / LF endings / YAML·TOML validation.

```bash
pip install -r requirements-dev.txt
pre-commit install          # install the git hook (runs on every commit)
pre-commit run --all-files  # manual full check
```

## Notes

- **Forward-only** (no autograd). For training/backward, register a backward implementation in `ops.cpp`, or wrap the op in a `torch.autograd.Function` on the Python side.
- The binding can be switched to plain **pybind11** (`cuda_gemm.ops.foo(...)`) by replacing the `TORCH_LIBRARY*` block in `ops.cpp` with `PYBIND11_MODULE` + `m.def_submodule("ops")`; the kernels and `setup.py` stay the same.
