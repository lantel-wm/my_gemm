#ifndef TEST_GEMM_HPP
#define TEST_GEMM_HPP

#include "gemm_ground_truth.hpp"
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <string>

using GEMM_FUNC = std::function<void(const float *, const float *, float *)>;

template <int M, int N, int K>
void test_gemm(GEMM_FUNC gemm, const std::string &gemm_name) {
  std::cout << "Testing " << gemm_name << " with dimensions: M=" << M
            << ", N=" << N << ", K=" << K << std::endl;
  // Allocate host memory
  float *h_A = new float[M * K];
  float *h_B = new float[K * N];
  float *h_C = new float[M * N];
  float *h_expected_C = new float[M * N];

  // Initialize input matrices with random values
  std::random_device rd;
  // Set a fixed seed for reproducible results
  std::mt19937 gen(42); // Using a fixed seed value of 42
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < M * K; i++) {
    h_A[i] = dist(gen);
  }
  for (int i = 0; i < K * N; i++) {
    h_B[i] = dist(gen);
  }

  // Compute expected output using CPU implementation
  gemm_ground_truth<M, N, K>(h_A, h_B, h_expected_C);

  // Call the GEMM implementation under test
  gemm(h_A, h_B, h_C);

  // Verify results
  const float epsilon = 1e-3f;
  bool pass = true;
  for (int i = 0; i < M * N; i++) {
    if (std::fabs(h_C[i] - h_expected_C[i]) > epsilon) {
      std::cout << "Error at index " << i << ": expected " << h_expected_C[i]
                << ", got " << h_C[i] << std::endl;
      pass = false;
      break;
    }
  }

  // Cleanup
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_expected_C;

  if (pass) {
    std::cout << gemm_name << " test PASSED!" << std::endl;
  } else {
    std::cout << gemm_name << " test FAILED!" << std::endl;
    assert(false);
  }
}

#endif // TEST_GEMM_HPP
