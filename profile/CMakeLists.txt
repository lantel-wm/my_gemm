cmake_minimum_required(VERSION 3.18)
project(gemm_profile CUDA CXX)

# Set CUDA compiler and flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA compiler flags for better profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # Device code debugging
endif()

# Include source directory for header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

# Add profile executables for all gemm versions
file(GLOB PROFILE_GEMM_FILES "profile_gemm_v*.cpp")

foreach(PROFILE_FILE ${PROFILE_GEMM_FILES})
    get_filename_component(PROFILE_NAME ${PROFILE_FILE} NAME_WE)
    add_executable(${PROFILE_NAME} 
        ${PROFILE_FILE}
        profile.cpp
    )
    
    # Enable CUDA compilation
    set_source_files_properties(${PROFILE_FILE} PROPERTIES LANGUAGE CUDA)
    
    # Link against the corresponding gemm library
    string(REGEX REPLACE "profile_(.*)" "\\1_lib" LIB_NAME ${PROFILE_NAME})
    target_link_libraries(${PROFILE_NAME} ${LIB_NAME} ${CUDA_LIBRARIES})
    
    # Set architecture-specific optimizations
    set_target_properties(${PROFILE_NAME} PROPERTIES 
        CUDA_ARCHITECTURES "89"  # RTX 4090 is Compute Capability 8.9
    )
endforeach()

# Add the executable for CUBLAS GEMM
add_executable(profile_cublas_gemm 
    profile_cublas_gemm.cpp
    profile.cpp
)

# Enable CUDA compilation for profile_cublas_gemm.cpp
set_source_files_properties(profile_cublas_gemm.cpp PROPERTIES LANGUAGE CUDA)

# Link against the cublas_gemm header-only library
target_link_libraries(profile_cublas_gemm cublas_gemm_lib ${CUDA_LIBRARIES})

# Set architecture-specific optimizations for CUBLAS
set_target_properties(profile_cublas_gemm PROPERTIES 
    CUDA_ARCHITECTURES "89"  # RTX 4090 is Compute Capability 8.9
)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif() 