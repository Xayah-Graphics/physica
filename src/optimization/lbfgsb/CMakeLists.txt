# ============================================================================
# L-BFGS-B
# ============================================================================
add_library(physica-optimization-lbfgsb)
target_sources(
        physica-optimization-lbfgsb
        PRIVATE
        lbfgsb-kernels.h
        lbfgsb.cpp
        lbfgsb.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        lbfgsb.ixx
)
target_include_directories(
        physica-optimization-lbfgsb
        PRIVATE
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-optimization-lbfgsb
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
)
