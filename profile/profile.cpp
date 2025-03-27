#include "profile.hpp"
#include <chrono>
#include <iostream>
#include <random>

void initialize_matrix(float *matrix, int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // Set a fixed seed for reproducible results
  gen.seed(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < size; ++i) {
    matrix[i] = dist(gen);
  }
}

void check_cuda(cudaError_t result, const char *func, const char *file,
                int line) {
  if (result) {
    std::cerr << "CUDA error at " << file << ":" << line
              << " code=" << static_cast<unsigned int>(result) << " \"" << func
              << "\" " << cudaGetErrorString(result) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void profile(int M, int N, int K, GEMM_FUNC gemm, int retry) {
  std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K
            << std::endl;

  // Allocate host memory
  float *h_input = new float[M * K];
  float *h_weight = new float[K * N];
  float *h_output = new float[M * N];

  // Initialize matrices with random values
  initialize_matrix(h_input, M * K);
  initialize_matrix(h_weight, K * N);

  // Warm-up run to initialize GPU
  gemm(h_input, h_weight, h_output);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  auto min_duration = std::chrono::duration<double, std::milli>::max();
  for (int i = 0; i < retry; i++) {
    // Time the kernel execution
    auto start = std::chrono::high_resolution_clock::now();

    // Run the GEMM implementation - this will be profiled by nsight compute
    gemm(h_input, h_weight, h_output);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    min_duration = std::min(min_duration, duration);
  }

  // Calculate and print statistics
  double elapsed_time = min_duration.count();
  std::cout << "Execution time: " << elapsed_time << " ms" << std::endl;

  // Calculate theoretical FLOPS
  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);        // 2 operations per multiply-add
  double gflops = flops / (elapsed_time * 1e6); // Convert to GFLOPS
  std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

  // Free host memory
  delete[] h_input;
  delete[] h_weight;
  delete[] h_output;
}
