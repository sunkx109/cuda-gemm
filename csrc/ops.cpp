#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <utility>  // std::move

#include "launchers.h"

// Shared validation + marshalling for every gemm variant. The only thing that
// differs between backends is the kernel launch, passed in here as a function
// pointer — so adding a new variant needs no copy of this logic.
static torch::Tensor gemm_impl(torch::Tensor A, torch::Tensor B, gemm_launcher_fn launch)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(
        A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
        "Inputs must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Shape mismatch: A.size(1) must equal B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    launch(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K,
        at::cuda::getCurrentCUDAStream());

    return C;
}

// One-line dispatchers: each variant is just gemm_impl + its backend launcher.
torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B)
{
    return gemm_impl(std::move(A), std::move(B), launch_matmul_naive);
}

torch::Tensor gemm_gmem_coalesce(torch::Tensor A, torch::Tensor B)
{
    return gemm_impl(std::move(A), std::move(B), launch_matmul_gmem_coalesce);
}

torch::Tensor gemm_smem(torch::Tensor A, torch::Tensor B)
{
    return gemm_impl(std::move(A), std::move(B), launch_matmul_gmem_smem);
}

torch::Tensor gemm_blocktiling_1d(torch::Tensor A, torch::Tensor B)
{
    return gemm_impl(std::move(A), std::move(B), launch_matmul_gemm_blocktiling_1d);
}

torch::Tensor gemm_blocktiling_2d(torch::Tensor A, torch::Tensor B)
{
    return gemm_impl(std::move(A), std::move(B), launch_matmul_gemm_blocktiling_2d);
}

torch::Tensor gemm_warptiling(torch::Tensor A, torch::Tensor B)
{
    return gemm_impl(std::move(A), std::move(B), launch_matmul_gemm_warptiling);
}
// Register the operator schemas with the PyTorch dispatcher (torchbind).
// After `import cuda_gemm`, these are callable as torch.ops.cuda_gemm.*.
TORCH_LIBRARY(cuda_gemm, m)
{
    m.def("gemm_naive(Tensor A, Tensor B) -> Tensor");
    m.def("gemm_smem(Tensor A, Tensor B) -> Tensor");
    m.def("gemm_gmem_coalesce(Tensor A, Tensor B) -> Tensor");
    m.def("gemm_blocktiling_1d(Tensor A, Tensor B) -> Tensor");
    m.def("gemm_blocktiling_2d(Tensor A, Tensor B) -> Tensor");
    m.def("gemm_warptiling(Tensor A, Tensor B) -> Tensor");
}

// Bind the wrappers above to the CUDA dispatcher key.
TORCH_LIBRARY_IMPL(cuda_gemm, CUDA, m)
{
    m.impl("gemm_naive", &gemm_naive);
    m.impl("gemm_gmem_coalesce", &gemm_gmem_coalesce);
    m.impl("gemm_smem", &gemm_smem);
    m.impl("gemm_blocktiling_1d", &gemm_blocktiling_1d);
    m.impl("gemm_blocktiling_2d", &gemm_blocktiling_2d);
    m.impl("gemm_warptiling", &gemm_warptiling);
}

// Minimal importable module: importing this .so triggers the static
// TORCH_LIBRARY registrations above so that torch.ops.cuda_gemm.* becomes
// available. The body is intentionally empty.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.doc() = "cuda_gemm custom operators"; }
