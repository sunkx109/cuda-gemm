#pragma once

// Raw-pointer CUDA launchers, decoupled from PyTorch.
//
// Kept torch-free on purpose: these translation units are compiled by nvcc,
// and pulling <torch/extension.h> into a .cu file forces nvcc to re-parse
// PyTorch's template-heavy headers every build (~85 s per file). With only
// <cuda_runtime.h>, each .cu compiles in well under a second. The torch-facing
// wrappers (validation + tensor marshalling) live in ops.cpp, compiled by g++.

#include <cuda_runtime.h>

// C = A @ B, all row-major float32.  A: [M, K], B: [K, N], C: [M, N].
// Every backend shares this exact signature so the torch wrapper can dispatch
// to any of them through a single function pointer (see gemm_launcher_fn).
void launch_matmul_naive(
    const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);

void launch_matmul_gmem_coalesce(
    const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);

void launch_matmul_gmem_smem(
    const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);

void launch_matmul_gemm_blocktiling_1d(
    const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);

void launch_matmul_gemm_blocktiling_2d(
    const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);

void launch_matmul_gemm_warptiling(
    const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);

// Pointer to any launch_matmul_* function. ops.cpp's shared wrapper takes one
// of these, so adding a backend is "write a .cu launcher + one dispatch line".
using gemm_launcher_fn =
    void (*)(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);
