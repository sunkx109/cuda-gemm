#include <cuda_runtime.h>

#include "launchers.h"

// Grid : {(M + BM - 1) / BM, (N + BN - 1) / BN}
// Block : (BM / TM) * BN
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktiling_1d_kernel(
    const float *matrix_a, const float *matrix_b, float *matrix_c, int num_rows_a, int num_cols_b,
    int num_cols_a, float alpha, float beta)
{
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    const uint thread_row = threadIdx.x / BN;
    const uint thread_col = threadIdx.x % BN;

    // Calculate global row and column indices for this thread
    const int global_row = block_row * BM + thread_row * TM;
    const int global_col = block_col * BN + thread_col;

    // Move pointers to the starting position for this block
    matrix_a += block_row * BM * num_cols_a;  // row=block_row, col=0
    matrix_b += block_col * BN;               // row=0, col=block_col
    matrix_c += block_row * BM * num_cols_b + block_col * BN;

    // Allocate thread-local cache for results in register file
    // Instead of single accumulator, we have TM accumulators per thread
    float thread_results[TM] = {0.0f};

    // Loop over all tiles along K dimension
    for (int tile_idx = 0; tile_idx < num_cols_a; tile_idx += BK)
    {
        // Load tile from matrix A into shared memory with bounds checking
        // Each thread loads one element from A
        const uint a_row = threadIdx.x / BK;
        const uint a_col = threadIdx.x % BK;
        if ((block_row * BM + a_row) < num_rows_a && (tile_idx + a_col) < num_cols_a)
        {
            tile_a[a_row * BK + a_col] = matrix_a[a_row * num_cols_a + a_col];
        }
        else
        {
            tile_a[a_row * BK + a_col] = 0.0f;
        }

        // Load tile from matrix B into shared memory with bounds checking
        // Each thread loads one element from B
        const uint b_row = threadIdx.x / BN;
        const uint b_col = threadIdx.x % BN;
        if ((tile_idx + b_row) < num_cols_a && (block_col * BN + b_col) < num_cols_b)
        {
            tile_b[b_row * BN + b_col] = matrix_b[b_row * num_cols_b + b_col];
        }
        else
        {
            tile_b[b_row * BN + b_col] = 0.0f;
        }

        __syncthreads();

        // Advance pointers to next tile
        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        // Calculate per-thread results
        // 完成[BM,BK] * [BK,BN] 这个tile的gemm
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
        {
            // We make the dotproduct loop the outside loop, which facilitates
            // reuse of the tile_b entry, which we can cache in a tmp var.
            float b_tmp = tile_b[dot_idx * BN + thread_col];
            for (uint res_idx = 0; res_idx < TM; ++res_idx)
            {
                thread_results[res_idx] +=
                    tile_a[(thread_row * TM + res_idx) * BK + dot_idx] * b_tmp;
            }
        }

        __syncthreads();
    }

    // Write results to global memory: C = α*(A@B)+β*C
    for (uint res_idx = 0; res_idx < TM; ++res_idx)
    {
        int row = global_row + res_idx;
        if (row < num_rows_a && global_col < num_cols_b)
        {
            matrix_c[(thread_row * TM + res_idx) * num_cols_b + thread_col] =
                alpha * thread_results[res_idx] +
                beta * matrix_c[(thread_row * TM + res_idx) * num_cols_b + thread_col];
        }
    }
}

void launch_matmul_gemm_blocktiling_1d(
    const float *A, const float *B, float *C, int M, int N, int K, cudaStream_t stream)
{
    // Template parameters for kernel
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;

    // Configure kernel launch
    // Number of threads = (BM / TM) * BN = (64 / 8) * 64 = 512 threads per block
    dim3 block((BM / TM) * BN);  // 1D block of block_size * block_size threads
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    // Launch kernel
    sgemm_blocktiling_1d_kernel<BM, BN, BK, TM>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K, 1.0f, 0.0f);
}
