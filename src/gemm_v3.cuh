// Global memory tile + register tile
#ifndef GEMM_V3_CUH
#define GEMM_V3_CUH

#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

template <int TILE_SIZE_M, int TILE_SIZE_N, int BLOCK_SIZE_M, int BLOCK_SIZE_N,
          int BLOCK_SIZE_K, int M, int N, int K>
__global__ void gemm_v3_kernel(const float *__restrict__ A,
                               const float *__restrict__ B,
                               float *__restrict__ C) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int k_A = tx % BLOCK_SIZE_K;
  const int k_B = ty % BLOCK_SIZE_K;
  const int row = ty * TILE_SIZE_M + by * BLOCK_SIZE_M;
  const int col = tx * TILE_SIZE_N + bx * BLOCK_SIZE_N;

  __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K];
  __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];

  float a[TILE_SIZE_M];
  float b[TILE_SIZE_N];
  float c[TILE_SIZE_M][TILE_SIZE_N] = {0};

  for (int k_start = 0; k_start < K; k_start += BLOCK_SIZE_K) {
#pragma unroll
    for (int i = 0; i < TILE_SIZE_M; i++) {
      if (row + i < M && k_A + k_start < K) {
        As[ty * TILE_SIZE_M + i][k_A] = A[OFFSET(row + i, k_A + k_start, K)];
      } else {
        As[ty * TILE_SIZE_M + i][k_A] = 0.0f;
      }

      if (col + i < N && k_B + k_start < K) {
        Bs[k_B][tx * TILE_SIZE_N + i] = B[OFFSET(k_B + k_start, col + i, N)];
      } else {
        Bs[k_B][tx * TILE_SIZE_N + i] = 0.0f;
      }
    }

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE_K; k++) {
      for (int i = 0; i < TILE_SIZE_M; i++) {
        a[i] = As[ty * TILE_SIZE_M + i][k];
        b[i] = Bs[k][tx * TILE_SIZE_N + i];
      }
      for (int i = 0; i < TILE_SIZE_M; i++) {
        for (int j = 0; j < TILE_SIZE_N; j++) {
          c[i][j] += a[i] * b[j];
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < TILE_SIZE_M; i++) {
    for (int j = 0; j < TILE_SIZE_N; j++) {
      if (row + i < M && col + j < N) {
        C[OFFSET(row + i, col + j, N)] = c[i][j];
      }
    }
  }
}

template <int TILE_SIZE_M, int TILE_SIZE_N, int BLOCK_SIZE_M, int BLOCK_SIZE_N,
          int BLOCK_SIZE_K, int M, int N, int K>
void gemm_v3(const float *A, const float *B, float *C) {
  // Allocate device memory
  float *d_input, *d_weight, *d_output;
  cudaMalloc(&d_input, M * K * sizeof(float));
  cudaMalloc(&d_weight, K * N * sizeof(float));
  cudaMalloc(&d_output, M * N * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_input, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  constexpr int num_blocks_m = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
  constexpr int num_blocks_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
  constexpr int block_size_x = BLOCK_SIZE_N / TILE_SIZE_N;
  constexpr int block_size_y = BLOCK_SIZE_M / TILE_SIZE_M;

  // Define block and grid dimensions
  // Each thread processes one element in the output matrix
  // Block dimensions must accommodate both BLOCK_SIZE_M and BLOCK_SIZE_K or
  // BLOCK_SIZE_N
  dim3 blockDim(block_size_x, block_size_y);
  dim3 gridDim(num_blocks_n, num_blocks_m);

  // Launch kernel
  gemm_v3_kernel<TILE_SIZE_M, TILE_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_N,
                 BLOCK_SIZE_K, M, N, K>
      <<<gridDim, blockDim>>>(d_input, d_weight, d_output);

  // Copy result back to host
  cudaMemcpy(C, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
}

#endif // GEMM_V3_CUH