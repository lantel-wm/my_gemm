#ifndef GEMM_GROUND_TRUTH_HPP
#define GEMM_GROUND_TRUTH_HPP

#include <cuda_runtime.h>

template <int M, int N, int K>
__global__ void gemm_ground_truth_kernel(const float *A, const float *B,
                                         float *C) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= M || j >= N) {
    return;
  }

  float sum = 0.0f;
  for (int k = 0; k < K; k++) {
    sum += A[i * K + k] * B[k * N + j];
  }

  C[i * N + j] = sum;
}

// GEMM for result verification
template <int M, int N, int K>
void gemm_ground_truth(const float *A, const float *B, float *C) {
  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  // Define block and grid dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y);

  // Launch kernel
  gemm_ground_truth_kernel<M, N, K><<<gridDim, blockDim>>>(d_A, d_B, d_C);

  // Copy result back to host
  cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

#endif // GEMM_GROUND_TRUTH_HPP