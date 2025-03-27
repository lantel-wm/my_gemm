file(GLOB GEMM_V_FILES "gemm_v*.cuh")

foreach(GEMM_FILE ${GEMM_V_FILES})
    get_filename_component(GEMM_NAME ${GEMM_FILE} NAME_WE)
    add_library(${GEMM_NAME}_lib INTERFACE)
    target_include_directories(${GEMM_NAME}_lib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endforeach()

add_library(cublas_gemm_lib INTERFACE)
target_include_directories(cublas_gemm_lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cublas_gemm_lib INTERFACE ${CUDA_LIBRARIES} cublas)


