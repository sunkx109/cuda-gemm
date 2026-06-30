#include <cuda_runtime.h>

#include "launchers.h"

// gridsize = {num_rows_a / block_size, num_cols_b / block_size}
// blocksize = {block_size * block_size}

template <const uint block_size>
__global__ void sgemm_shared_mem_kernel(const float *matrix_a,
                                        const float *matrix_b,float *matrix_c,int num_rows_a, int num_cols_b, int num_cols_a,
                                        float alpha, float beta)
{
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    __shared__ float tile_a[block_size * block_size];
    __shared__ float tile_b[block_size * block_size];

    // 一个block内，一共有block_size*block_size个线程，每个线程负责计算一个C矩阵的元素
    // 定义了块smem，每个线程分别去load a 和 b的tile块

    const uint thread_row = threadIdx.x / block_size;
    const uint thread_col = threadIdx.x % block_size;

    // Calculate global row and column indices for this thread
    const uint global_row = block_row * block_size + thread_row;
    const uint global_col = block_col * block_size + thread_col;

    // Move pointers to the starting position for this block
    matrix_a += block_row * block_size * num_cols_a; // row=block_row, col=0
    matrix_b += block_col * block_size;              // row=0, col=block_col
    matrix_c += block_row * block_size * num_cols_b + block_col * block_size;

    float accumulator = 0.0f;

    // Loop over all tiles along K dimension
    for (int tile_idx = 0; tile_idx < num_cols_a; tile_idx += block_size)
    {
        // Load tile from matrix A into shared memory with bounds checking
        // thread_col is consecutive for coalesced memory access
        if (global_row < num_rows_a && (tile_idx + thread_col) < num_cols_a)
        {
            tile_a[thread_row * block_size + thread_col] =
                matrix_a[thread_row * num_cols_a + thread_col];
        }
        else
        {
            tile_a[thread_row * block_size + thread_col] = 0.0f;
        }

        // Load tile from matrix B into shared memory with bounds checking
        // thread_col is consecutive for coalesced memory access
        if ((tile_idx + thread_row) < num_cols_a && global_col < num_cols_b)
        {
            tile_b[thread_row * block_size + thread_col] =
                matrix_b[thread_row * num_cols_b + thread_col];
        }
        else
        {
            //其余部分补0
            tile_b[thread_row * block_size + thread_col] = 0.0f;
        }

        // Block threads until cache is fully populated
        __syncthreads();

        // Advance pointers to next tile
        matrix_a += block_size;
        matrix_b += block_size * num_cols_b;

        // Compute partial dot product using shared memory
        for (int dot_idx = 0; dot_idx < block_size; ++dot_idx)
        {
            accumulator += tile_a[thread_row * block_size + dot_idx] *
                           tile_b[dot_idx * block_size + thread_col];
        }

        // Sync again to avoid faster threads fetching next block before slower threads finish
        __syncthreads();
    }

    // Write result to global memory with bounds checking: C = α*(A@B)+β*C
    if (global_row < num_rows_a && global_col < num_cols_b)
    {
        matrix_c[thread_row * num_cols_b + thread_col] =
            alpha * accumulator + beta * matrix_c[thread_row * num_cols_b + thread_col];
    }
}

void launch_matmul_gmem_smem(
    const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)
{
    constexpr int BLOCK = 32;
    dim3 block(BLOCK * BLOCK);  // 1D block of block_size * block_size threads
    dim3 grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

    // Launch kernel
    sgemm_shared_mem_kernel<BLOCK><<<grid, block, 0, stream>>>(A, B, C,M, N, K,1.0f,0.0f);
}