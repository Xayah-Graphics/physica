# ============================================================================
# Liquid Domain
# ============================================================================
add_library(physica-fluids-liquid-domain)
target_sources(
        physica-fluids-liquid-domain
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
        physica-fluids-liquid-domain
        PUBLIC
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-fluids-liquid-domain
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
)

# ============================================================================
# Liquid Operators
# ============================================================================
add_library(physica-fluids-liquid-operators)
target_sources(
        physica-fluids-liquid-operators
        PRIVATE
        operators/density-kernels.h
        operators/density.cu
        operators/neighborhood-kernels.h
        operators/neighborhood.cpp
        operators/neighborhood.cu
        operators/sph-dynamics-common.cu
        operators/sph-dynamics-dfsph.cu
        operators/sph-dynamics-iisph.cu
        operators/sph-dynamics-kernels.h
        operators/sph-dynamics-pcisph.cu
        operators/sph-dynamics-wcsph.cu
        operators/sph-dynamics.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        operators/density.ixx
        operators/neighborhood.ixx
        operators/sph-dynamics.ixx
)
target_include_directories(
        physica-fluids-liquid-operators
        PRIVATE
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-fluids-liquid-operators
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-fluids-liquid-domain
)

# ============================================================================
# SPH Solver
# ============================================================================
add_library(physica-fluids-liquid-sph)
target_sources(
        physica-fluids-liquid-sph
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        solvers/sph/sph.ixx
)
target_include_directories(
        physica-fluids-liquid-sph
        PUBLIC
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-fluids-liquid-sph
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-fluids-liquid-domain
        physica-fluids-liquid-operators
)

# ============================================================================
# PBF Solver
# ============================================================================
add_library(physica-fluids-liquid-pbf)
target_sources(
        physica-fluids-liquid-pbf
        PRIVATE
        solvers/pbf/pbf-kernels.h
        solvers/pbf/pbf.cpp
        solvers/pbf/pbf.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        solvers/pbf/pbf.ixx
)
target_include_directories(
        physica-fluids-liquid-pbf
        PRIVATE
        "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        physica-fluids-liquid-pbf
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-fluids-liquid-domain
        physica-fluids-liquid-operators
)
