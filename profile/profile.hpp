#ifndef PROFILE_HPP
#define PROFILE_HPP

#include <cuda_runtime.h>
#include <functional>

// Helper function to initialize matrices with random values
void initialize_matrix(float *matrix, int size);

// Helper function to check CUDA errors
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, const char *func, const char *file,
                int line);

using GEMM_FUNC = std::function<void(const float *, const float *, float *)>;

void profile(int M, int N, int K, GEMM_FUNC gemm, int retry = 100);

#endif // PROFILE_HPP