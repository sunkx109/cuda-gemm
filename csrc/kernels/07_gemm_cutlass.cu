#include <cuda_runtime.h>
  #include <cutlass/cutlass.h>
  #include <cutlass/gemm/device/gemm.h>

  #include "launchers.h"

  // 定义一个基于 SIMT 的 GEMM kernel（兼容性最广，不依赖 Tensor Cores）
  using CutlassGemm = cutlass::gemm::device::Gemm<
      float,                              // ElementA
      cutlass::layout::RowMajor,          // LayoutA
      float,                              // ElementB
      cutlass::layout::RowMajor,          // LayoutB
      float,                              // ElementC
      cutlass::layout::RowMajor,          // LayoutC
      float,                              // ElementAccumulator
      cutlass::arch::OpClassSimt,         // SIMT (不用 Tensor Core)
      cutlass::arch::Sm120,
      cutlass::gemm::GemmShape<128, 128, 8>,   // Threadblock tile: 128x128x8
      cutlass::gemm::GemmShape<32, 64, 8>,     // Warp tile: 32x64x8
      cutlass::gemm::GemmShape<1, 1, 1>,       // Instruction: 1x1x1 (scalar SIMT)
      cutlass::epilogue::thread::LinearCombination<
          float,                          // ElementOutput
          1,                              // ElementsPerAccess
          float,                          // ElementAccumulator
          float                           // ElementCompute
      >,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      2                                   // NumStages (shared memory pipeline depth)
  >;

  void launch_matmul_gemm_cutlass(
      const float* A, const float* B, float* C,
      int M, int N, int K, cudaStream_t stream)
  {
      CutlassGemm gemm_op;

      // CUTLASS 2.x 用 Arguments 结构体传参
      // GemmCoord{M, N, K}, TensorRef{A, lda}, TensorRef{B, ldb}, ...
      typename CutlassGemm::Arguments args{
          {M, N, K},          // problem size
          {A, K},             // A: MxK row-major, leading dim = K
          {B, N},             // B: KxN row-major, leading dim = N
          {C, N},             // C (initial): MxN row-major, leading dim = N
          {C, N},             // D (output): same as C
          {1.0f, 0.0f}        // alpha, beta  →  C = 1.0 * A * B + 0.0 * C
      };

      cutlass::Status status = gemm_op(args);

      if (status != cutlass::Status::kSuccess) {
          // 这里不能抛异常（.cu 编译没有 RTTI），简单打印即可
          printf("CUTLASS GEMM failed with status %d\n", static_cast<int>(status));
      }
  }