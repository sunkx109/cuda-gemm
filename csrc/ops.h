#pragma once

#include <torch/extension.h>

// Computes C = A @ B for 2D float32 CUDA tensors.
//   A: [M, K],  B: [K, N]  ->  C: [M, N]
torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B);
