// Global memory tile
#ifndef GEMM_V2_CUH
#define GEMM_V2_CUH

#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int M, int N,
          int K>
__global__ void gemm_v2_kernel(const float *__restrict__ A,
                               const float *__restrict__ B,
                               float *__restrict__ C) {
  const int j = threadIdx.x;
  const int i = threadIdx.y;
  const int k_A = j % BLOCK_SIZE_K;
  const int k_B = i % BLOCK_SIZE_K;
  const int row = threadIdx.y + blockIdx.y * blockDim.y;
  const int col = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K];
  __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];

  float sum = 0.0f;

  for (int k_start = 0; k_start < K; k_start += BLOCK_SIZE_K) {
    if (row < M && k_A + k_start < K) {
      As[i][k_A] = A[OFFSET(row, k_A + k_start, K)];
    } else {
      As[i][k_A] = 0.0f;
    }

    if (col < N && k_B + k_start < K) {
      Bs[k_B][j] = B[OFFSET(k_B + k_start, col, N)];
    } else {
      Bs[k_B][j] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE_K; k++) {
      sum += As[i][k] * Bs[k][j];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[OFFSET(row, col, N)] = sum;
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int M, int N,
          int K>
void gemm_v2(const float *A, const float *B, float *C) {
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

  // Define block and grid dimensions
  // Each thread processes one element in the output matrix
  // Block dimensions must accommodate both BLOCK_SIZE_M and BLOCK_SIZE_K or
  // BLOCK_SIZE_N
  dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M);
  dim3 gridDim(num_blocks_n, num_blocks_m);

  // Launch kernel
  gemm_v2_kernel<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, M, N, K>
      <<<gridDim, blockDim>>>(d_input, d_weight, d_output);

  // Copy result back to host
  cudaMemcpy(C, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
}

#endif // GEMM_V2_CUH