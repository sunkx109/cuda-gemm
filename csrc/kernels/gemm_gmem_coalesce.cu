#include <cuda_runtime.h>

#include "launchers.h"

// Naive GEMM: one thread per output element C[row][col].
// Grid  [M/block_size , N/block_size]
// Block [block_size * block_size]
template <const int block_size>
__global__ void matmul_gmem_coalesce_kernel(
    const float* matrix_a, const float* matrix_b, float* output_matrix, int num_rows_a,
    int num_cols_b, int num_cols_a, float alpha, float beta)
{
    // Map 1D thread ID to 2D output position
    const int output_row = blockIdx.x * block_size + (threadIdx.x / block_size);
    const int output_col = blockIdx.y * block_size + (threadIdx.x % block_size);

    // Boundary check for non-multiple of block size
    if (output_row < num_rows_a && output_col < num_cols_b)
    {
        float accumulator = 0.0f;
        for (int k_idx = 0; k_idx < num_cols_a; ++k_idx)
        {
            accumulator += matrix_a[output_row * num_cols_a + k_idx] *
                           matrix_b[k_idx * num_cols_b + output_col];
        }
        // C = α*(A@B)+β*C
        const int output_idx = output_row * num_cols_b + output_col;
        output_matrix[output_idx] = alpha * accumulator + beta * output_matrix[output_idx];
    }
}

void launch_matmul_gmem_coalesce(
    const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)
{
    constexpr int BLOCK = 32;
    dim3 block(BLOCK * BLOCK);  // 1D block of block_size * block_size threads
    dim3 grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

    matmul_gmem_coalesce_kernel<BLOCK><<<grid, block, 0, stream>>>(A, B, C, M, N, K, 1.0f, 0.0f);
}
