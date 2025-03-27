cmake_minimum_required(VERSION 3.18)
project(gemm_rtx4090 CUDA CXX)

# Enable compile commands export for IDE integration and tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CUDA compiler and flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set common CUDA flags for all targets
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code=arch=compute_89,code=[compute_89,sm_89]")

# Find CUDA and enable CUDA language
find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})

# Add source files
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)
set(PROFILE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/profile)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Add subdirectories
add_subdirectory(${SRC_DIR})
add_subdirectory(${TEST_DIR})
add_subdirectory(${PROFILE_DIR}) 