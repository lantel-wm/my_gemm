project(gemm_test CUDA CXX)

# Set CUDA compiler and flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create test executable
add_executable(test_gemm test_gemm.cpp)
# Enable CUDA compilation
set_source_files_properties(test_gemm.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(test_gemm gemm_v1_lib ${CUDA_LIBRARIES})
target_link_libraries(test_gemm gemm_v2_lib ${CUDA_LIBRARIES})
target_link_libraries(test_gemm gemm_v3_lib ${CUDA_LIBRARIES})
target_link_libraries(test_gemm gemm_v4_lib ${CUDA_LIBRARIES})
target_link_libraries(test_gemm gemm_v5_lib ${CUDA_LIBRARIES})
target_link_libraries(test_gemm cublas_gemm_lib ${CUDA_LIBRARIES})

# Enable architecture-specific optimizations for test_gemm
set_target_properties(test_gemm PROPERTIES CUDA_ARCHITECTURES "89")  # RTX 4090 is Compute Capability 8.9 