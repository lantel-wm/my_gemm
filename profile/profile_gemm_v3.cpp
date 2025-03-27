#include "../src/gemm_v3.cuh"
#include "profile.hpp"

int main(int argc, char *argv[]) {
  // Default matrix dimensions
  constexpr int M = 1024; // Rows of input matrix (seq_len)
  constexpr int N = 3584; // Columns of weight matrix (hidden_size)
  constexpr int K = 3584; // Columns of input / rows of weight (hidden_size)
  constexpr int BLOCK_SIZE_M = 64;
  constexpr int BLOCK_SIZE_N = 64;
  constexpr int BLOCK_SIZE_K = 16;
  constexpr int TILE_SIZE_M = 4;
  constexpr int TILE_SIZE_N = 4;

  profile(M, N, K,
          gemm_v3<TILE_SIZE_M, TILE_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_N,
                  BLOCK_SIZE_K, M, N, K>);

  return 0;
}