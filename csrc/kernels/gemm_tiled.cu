#include <cuda_runtime.h>

#include "launchers.h"

// Shared-memory tiled GEMM. Each thread block computes a TILE x TILE tile of C.
// Works for arbitrary (non-tile-aligned) dimensions via bounds checking.
constexpr int TILE = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    const int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t)
    {
        // Collaboratively load one tile from A and B into shared memory.
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k)
        {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = acc;
    }
}

void launch_matmul_tiled(
    const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
