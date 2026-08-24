# ============================================================================
# Cloth Solver
# ============================================================================
add_library(physica-deformables-cloth)
target_sources(
        physica-deformables-cloth
        PRIVATE
        domain/domain.cpp
        domain/domain.cu
        domain/device.h
        domain/device.cuh
        domain/interop.h
        domain/kernels.h
        forces/forces.cpp
        forces/forces.cu
        forces/kernels.h
        integration/integration.cpp
        integration/integration.cu
        integration/kernels.h
        constraints/constraints.cpp
        constraints/constraints.cu
        constraints/kernels.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        cloth.ixx
        domain/domain.ixx
        forces/forces.ixx
        integration/integration.ixx
        constraints/constraints.ixx
)
target_include_directories(
        physica-deformables-cloth
        PUBLIC
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-deformables-cloth
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
)
