#include "test_gemm.hpp"
#include "../src/cublas_gemm.cuh"
#include "../src/gemm_v1.cuh"
#include "../src/gemm_v2.cuh"
#include "../src/gemm_v3.cuh"
#include "../src/gemm_v4.cuh"
#include "../src/gemm_v5.cuh"

int main() {
  // Test dimensions
  // Default matrix dimensions
  // constexpr int M = 128; // Rows of input matrix (seq_len)
  // constexpr int N = 256; // Columns of weight matrix (hidden_size)
  // constexpr int K = 64;  // Columns of input / rows of weight (hidden_size)
  constexpr int M = 1024; // Rows of input matrix (seq_len)
  constexpr int N = 3584; // Columns of weight matrix (hidden_size)
  constexpr int K = 3584; // Columns of input / rows of weight (hidden_size)

  // Test GEMM v1
  test_gemm<M, N, K>(gemm_v1<M, N, K>, "GEMM v1");

  // Test GEMM v2
  test_gemm<M, N, K>(gemm_v2<16, 16, 16, M, N, K>, "GEMM v2");

  // Test GEMM v3
  test_gemm<M, N, K>(gemm_v3<4, 4, 64, 64, 16, M, N, K>, "GEMM v3");

  // Test GEMM v4
  test_gemm<M, N, K>(gemm_v4<4, 4, 64, 64, 16, M, N, K>, "GEMM v4");

  // Test GEMM v5
  test_gemm<M, N, K>(gemm_v5<4, 4, 64, 64, 16, M, N, K>, "GEMM v5");

  // Test CUBLAS GEMM
  test_gemm<M, N, K>(cublas_gemm<M, N, K>, "CUBLAS GEMM");

  return 0;
}
