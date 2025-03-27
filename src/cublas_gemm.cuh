#ifndef CUBLAS_GEMM_CUH
#define CUBLAS_GEMM_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>

template <int M, int N, int K>
void cublas_gemm(const float *A, const float *B, float *C) {
  // Create CUBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  // Set up CUBLAS parameters
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Perform matrix multiplication using CUBLAS
  // C = alpha * A * B + beta * C
  cublasSgemm(handle,
              CUBLAS_OP_N, // Operation for A matrix
              CUBLAS_OP_N, // Operation for B matrix
              N,           // Number of columns of matrix A and C
              M,           // Number of rows of matrix A and C
              K, // Number of columns of matrix B and number of rows of matrix A
              &alpha, // Scalar multiplier for A*B
              d_B,    // Matrix B on device
              N,      // Leading dimension of matrix B
              d_A,    // Matrix A on device
              K,      // Leading dimension of matrix A
              &beta,  // Scalar multiplier for C
              d_C,    // Matrix C on device
              N);     // Leading dimension of matrix C

  // Copy result back to host
  cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Destroy CUBLAS handle
  cublasDestroy(handle);
}

#endif // CUBLAS_GEMM_CUH
