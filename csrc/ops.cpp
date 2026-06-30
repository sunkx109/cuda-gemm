#include "ops.h"

#include <torch/library.h>

// Register the operator schemas with the PyTorch dispatcher (torchbind).
// After `import cuda_gemm`, these are callable as torch.ops.cuda_gemm.*.
TORCH_LIBRARY(cuda_gemm, m)
{
    m.def("gemm_naive(Tensor A, Tensor B) -> Tensor");
    m.def("gemm_tiled(Tensor A, Tensor B) -> Tensor");
}

// Bind the CUDA implementations (defined in csrc/gemm_*.cu).
TORCH_LIBRARY_IMPL(cuda_gemm, CUDA, m)
{
    m.impl("gemm_naive", &gemm_naive);
    m.impl("gemm_tiled", &gemm_tiled);
}

// Minimal importable module: importing this .so triggers the static
// TORCH_LIBRARY registrations above so that torch.ops.cuda_gemm.* becomes
// available. The body is intentionally empty.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.doc() = "cuda_gemm custom operators"; }
