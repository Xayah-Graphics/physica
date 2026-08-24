# ============================================================================
# Smoke Solver
# ============================================================================
add_library(physica-fluids-gas-smoke)
target_sources(
        physica-fluids-gas-smoke
        PRIVATE
        domain/domain.cpp
        domain/domain.cu
        domain/device.h
        domain/device.cuh
        domain/interop.h
        domain/kernels.h
        transport/transport.cpp
        transport/transport.cu
        transport/kernels.h
        transport/sampling.cuh
        forces/forces.cpp
        forces/forces.cu
        forces/kernels.h
        projection/projection.cpp
        projection/projection.cu
        projection/red_black_gauss_seidel.cu
        projection/kernels.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        smoke.ixx
        domain/domain.ixx
        transport/transport.ixx
        forces/forces.ixx
        projection/projection.ixx
)
target_include_directories(
        physica-fluids-gas-smoke
        PUBLIC
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-fluids-gas-smoke
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
)
