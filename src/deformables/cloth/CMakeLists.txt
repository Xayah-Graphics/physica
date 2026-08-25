# ============================================================================
# Cloth Domain
# ============================================================================
add_library(physica-deformables-cloth-domain)
target_sources(
        physica-deformables-cloth-domain
        PRIVATE
        detail/cuda/device.cuh
        detail/cuda/interop.h
        detail/cuda/types.h
        domain/domain-kernels.h
        domain/domain.cpp
        domain/domain.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        domain/domain.ixx
)
target_include_directories(
        physica-deformables-cloth-domain
        PUBLIC
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-deformables-cloth-domain
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
)

# ============================================================================
# Cloth Operators
# ============================================================================
add_library(physica-deformables-cloth-operators)
target_sources(
        physica-deformables-cloth-operators
        PRIVATE
        operators/fixed-position-kernels.h
        operators/fixed-position.cpp
        operators/fixed-position.cu
        operators/mass-spring-kernels.h
        operators/mass-spring.cpp
        operators/mass-spring.cu
        operators/semi-implicit-euler-kernels.h
        operators/semi-implicit-euler.cpp
        operators/semi-implicit-euler.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        operators/fixed-position.ixx
        operators/mass-spring.ixx
        operators/semi-implicit-euler.ixx
)
target_include_directories(
        physica-deformables-cloth-operators
        PRIVATE
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-deformables-cloth-operators
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-deformables-cloth-domain
)

# ============================================================================
# Explicit Dynamics Solver
# ============================================================================
add_library(physica-deformables-cloth-explicit-dynamics)
target_sources(
        physica-deformables-cloth-explicit-dynamics
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        solvers/explicit-dynamics/explicit-dynamics.ixx
)
target_include_directories(
        physica-deformables-cloth-explicit-dynamics
        PUBLIC
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-deformables-cloth-explicit-dynamics
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-deformables-cloth-domain
)
