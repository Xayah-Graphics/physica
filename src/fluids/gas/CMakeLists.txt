# ============================================================================
# Gas Fluids
# ============================================================================
add_library(physica-fluids-gas-domain)
target_sources(
        physica-fluids-gas-domain
        PRIVATE
        domain/domain.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        domain/domain.ixx
)
target_include_directories(
        physica-fluids-gas-domain
        PUBLIC
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-fluids-gas-domain
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
)

# ============================================================================
# Gas Operators
# ============================================================================
add_library(physica-fluids-gas-operators)
target_sources(
        physica-fluids-gas-operators
        PRIVATE
        operators/advection-kernels.h
        operators/advection-sampling.cuh
        operators/advection.cpp
        operators/advection.cu
        operators/conservation-kernels.h
        operators/conservation.cpp
        operators/conservation.cu
        operators/diffusion-kernels.h
        operators/diffusion.cpp
        operators/diffusion.cu
        operators/force-kernels.h
        operators/force.cpp
        operators/force.cu
        operators/objective-kernels.h
        operators/objective.cpp
        operators/objective.cu
        operators/pointwise-kernels.h
        operators/pointwise.cpp
        operators/pointwise.cu
        operators/projection-kernels.h
        operators/projection-red-black-gauss-seidel.cu
        operators/projection.cpp
        operators/projection.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        operators/advection.ixx
        operators/conservation.ixx
        operators/diffusion.ixx
        operators/force.ixx
        operators/objective.ixx
        operators/pointwise.ixx
        operators/projection.ixx
)
target_include_directories(
        physica-fluids-gas-operators
        PRIVATE
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-fluids-gas-operators
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-fluids-gas-domain
)

# ============================================================================
# Smoke Solver
# ============================================================================
add_library(physica-fluids-gas-smoke)
target_sources(
        physica-fluids-gas-smoke
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        solvers/smoke/smoke.ixx
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
        physica-fluids-gas-domain
        physica-fluids-gas-operators
)

# ============================================================================
# Keyframe Smoke Solver
# ============================================================================
add_library(physica-fluids-gas-keyframe-smoke)
target_sources(
        physica-fluids-gas-keyframe-smoke
        PRIVATE
        solvers/keyframe-smoke/control-kernels.h
        solvers/keyframe-smoke/control.cpp
        solvers/keyframe-smoke/control.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        solvers/keyframe-smoke/control.ixx
        solvers/keyframe-smoke/evaluation.ixx
        solvers/keyframe-smoke/keyframe-smoke.ixx
        solvers/keyframe-smoke/multiple-shooting.ixx
        solvers/keyframe-smoke/optimization.ixx
)
target_include_directories(
        physica-fluids-gas-keyframe-smoke
        PUBLIC
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-fluids-gas-keyframe-smoke
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-fluids-gas-domain
        physica-fluids-gas-operators
        physica-optimization-lbfgsb
)

# ============================================================================
# Adjoint Control Solver
# ============================================================================
add_library(physica-fluids-gas-adjoint-control)
target_sources(
        physica-fluids-gas-adjoint-control
        PRIVATE
        solvers/adjoint-control/control-kernels.h
        solvers/adjoint-control/control.cpp
        solvers/adjoint-control/control.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        solvers/adjoint-control/adjoint-control.ixx
        solvers/adjoint-control/control.ixx
        solvers/adjoint-control/evaluation.ixx
        solvers/adjoint-control/optimization.ixx
)
target_include_directories(
        physica-fluids-gas-adjoint-control
        PUBLIC
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-fluids-gas-adjoint-control
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-fluids-gas-domain
        physica-fluids-gas-operators
        physica-optimization-lbfgsb
)
