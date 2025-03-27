// Global memory tile + register tile + vectorized fetch and load
// + double buffer
#ifndef GEMM_V5_CUH
#define GEMM_V5_CUH

#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <int TILE_SIZE_M, int TILE_SIZE_N, int BLOCK_SIZE_M, int BLOCK_SIZE_N,
          int BLOCK_SIZE_K, int M, int N, int K>
__global__ void gemm_v5_kernel(float *__restrict__ A, float *__restrict__ B,
                               float *__restrict__ C) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int k_A = tx % BLOCK_SIZE_K;
  const int k_B = ty % BLOCK_SIZE_K;
  const int row = ty * TILE_SIZE_M + by * BLOCK_SIZE_M;
  const int col = tx * TILE_SIZE_N + bx * BLOCK_SIZE_N;

  __shared__ float As[2][BLOCK_SIZE_M][BLOCK_SIZE_K];
  __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

  float a[TILE_SIZE_M];
  float b[TILE_SIZE_N];
  float c[TILE_SIZE_M][TILE_SIZE_N] = {0};
  int load_stage = 0;
  int store_stage = 1;

// load first block
#pragma unroll
  for (int i = 0; i < TILE_SIZE_M; i += 4) {
    if (row + i < M) {
      As[load_stage][ty * TILE_SIZE_M + i][k_A] = A[OFFSET(row + i, k_A, K)];
      As[load_stage][ty * TILE_SIZE_M + i + 1][k_A] =
          A[OFFSET(row + i + 1, k_A, K)];
      As[load_stage][ty * TILE_SIZE_M + i + 2][k_A] =
          A[OFFSET(row + i + 2, k_A, K)];
      As[load_stage][ty * TILE_SIZE_M + i + 3][k_A] =
          A[OFFSET(row + i + 3, k_A, K)];
    }
    if (col + i < N) {
      FETCH_FLOAT4(Bs[load_stage][k_B][tx * TILE_SIZE_N + i]) =
          FETCH_FLOAT4(B[OFFSET(k_B, col + i, N)]);
    }
  }

  __syncthreads();

  for (int k_start = BLOCK_SIZE_K; k_start < K; k_start += BLOCK_SIZE_K) {
#pragma unroll
    for (int i = 0; i < TILE_SIZE_M; i += 4) {
      if (row + i < M && k_A + k_start < K) {
        As[store_stage][ty * TILE_SIZE_M + i][k_A] =
            A[OFFSET(row + i, k_A + k_start, K)];
        As[store_stage][ty * TILE_SIZE_M + i + 1][k_A] =
            A[OFFSET(row + i + 1, k_A + k_start, K)];
        As[store_stage][ty * TILE_SIZE_M + i + 2][k_A] =
            A[OFFSET(row + i + 2, k_A + k_start, K)];
        As[store_stage][ty * TILE_SIZE_M + i + 3][k_A] =
            A[OFFSET(row + i + 3, k_A + k_start, K)];
      }

      if (col + i < N && k_B + k_start < K) {
        FETCH_FLOAT4(Bs[store_stage][k_B][tx * TILE_SIZE_N + i]) =
            FETCH_FLOAT4(B[OFFSET(k_B + k_start, col + i, N)]);
      }
    }

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE_K; k++) {
#pragma unroll
      for (int i = 0; i < TILE_SIZE_M; i += 4) {
        a[i] = As[load_stage][ty * TILE_SIZE_M + i][k];
        a[i + 1] = As[load_stage][ty * TILE_SIZE_M + i + 1][k];
        a[i + 2] = As[load_stage][ty * TILE_SIZE_M + i + 2][k];
        a[i + 3] = As[load_stage][ty * TILE_SIZE_M + i + 3][k];
        FETCH_FLOAT4(b[i]) =
            FETCH_FLOAT4(Bs[load_stage][k][tx * TILE_SIZE_N + i]);
      }
#pragma unroll
      for (int i = 0; i < TILE_SIZE_M; i++) {
#pragma unroll
        for (int j = 0; j < TILE_SIZE_N; j++) {
          c[i][j] += a[i] * b[j];
        }
      }
    }

    store_stage = 1 - store_stage;
    load_stage = 1 - load_stage;
    __syncthreads();
  }

// compute last block
#pragma unroll
  for (int k = 0; k < BLOCK_SIZE_K; k++) {
#pragma unroll
    for (int i = 0; i < TILE_SIZE_M; i += 4) {
      a[i] = As[load_stage][ty * TILE_SIZE_M + i][k];
      a[i + 1] = As[load_stage][ty * TILE_SIZE_M + i + 1][k];
      a[i + 2] = As[load_stage][ty * TILE_SIZE_M + i + 2][k];
      a[i + 3] = As[load_stage][ty * TILE_SIZE_M + i + 3][k];
      FETCH_FLOAT4(b[i]) =
          FETCH_FLOAT4(Bs[load_stage][k][tx * TILE_SIZE_N + i]);
    }
#pragma unroll
    for (int i = 0; i < TILE_SIZE_M; i++) {
#pragma unroll
      for (int j = 0; j < TILE_SIZE_N; j++) {
        c[i][j] += a[i] * b[j];
      }
    }
  }

#pragma unroll
  for (int i = 0; i < TILE_SIZE_M; i++) {
#pragma unroll
    for (int j = 0; j < TILE_SIZE_N; j += 4) {
      if (row + i < M && col + j < N) {
        FETCH_FLOAT4(C[OFFSET(row + i, col + j, N)]) = FETCH_FLOAT4(c[i][j]);
      }
    }
  }
}

template <int TILE_SIZE_M, int TILE_SIZE_N, int BLOCK_SIZE_M, int BLOCK_SIZE_N,
          int BLOCK_SIZE_K, int M, int N, int K>
void gemm_v5(const float *A, const float *B, float *C) {
  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

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
  gemm_v5_kernel<TILE_SIZE_M, TILE_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_N,
                 BLOCK_SIZE_K, M, N, K><<<gridDim, blockDim>>>(d_A, d_B, d_C);

  // Copy result back to host
  cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

#endif // GEMM_V5_CUH