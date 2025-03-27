#include "../src/cublas_gemm.cuh"
#include "profile.hpp"

int main(int argc, char *argv[]) {
  // Default matrix dimensions
  constexpr int M = 1024; // Rows of input matrix (seq_len)
  constexpr int N = 3584; // Columns of weight matrix (hidden_size)
  constexpr int K = 3584; // Columns of input / rows of weight (hidden_size)

  profile(M, N, K, cublas_gemm<M, N, K>);

  return 0;
}