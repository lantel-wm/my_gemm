#include "../src/gemm_v2.cuh"
#include "profile.hpp"

int main(int argc, char *argv[]) {
  // Default matrix dimensions
  constexpr int M = 1024; // Rows of input matrix (seq_len)
  constexpr int N = 3584; // Columns of weight matrix (hidden_size)
  constexpr int K = 3584; // Columns of input / rows of weight (hidden_size)
  constexpr int BLOCK_SIZE_M = 16;
  constexpr int BLOCK_SIZE_N = 16;
  constexpr int BLOCK_SIZE_K = 16;

  profile(M, N, K, gemm_v2<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, M, N, K>);

  return 0;
}