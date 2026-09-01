# ============================================================================
# Project
# ============================================================================
cmake_minimum_required(VERSION 4.4)
set(CACHE{CMAKE_CUDA_ARCHITECTURES}
        TYPE STRING
        HELP "CUDA architectures"
        FORCE
        VALUE 120
)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" PHYSICA_VERSION LIMIT_COUNT 1)
project(physica VERSION "${PHYSICA_VERSION}" LANGUAGES CXX CUDA)

# ============================================================================
# Project Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_MODULE_STD ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PHYSICA_EXAMPLES "Build examples" ON)
option(PHYSICA_BUILD_SPECTRA "Build Spectra providers for examples" ON)
include(CTest)

# ============================================================================
# Dependencies
# ============================================================================
include(cmake/Dependencies.cmake)
physica_require_dependency(cuda)
physica_require_dependency(cudss)
physica_require_dependency(cccl)
physica_require_dependency(nlohmann_json)
physica_require_dependency(stb)
physica_require_dependency(ffmpeg)
include(cmake/ClionCudaFacade.cmake)

# ============================================================================
# Compiler Configuration
# ============================================================================
if (WIN32)
    add_compile_definitions(NOMINMAX)
endif ()

if (MSVC)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()

# ============================================================================
# Math
# ============================================================================
add_library(physica-math STATIC)
target_sources(
        physica-math
        PUBLIC
        FILE_SET headers TYPE HEADERS
        BASE_DIRS "${PROJECT_SOURCE_DIR}/physica"
        FILES physica/math/math.h
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/math/math.ixx
)
target_link_libraries(physica-math PUBLIC CCCL::libcudacxx)

# ============================================================================
# Optimization / L-BFGS-B
# ============================================================================
add_library(physica-optimization-lbfgsb STATIC)
target_sources(
        physica-optimization-lbfgsb
        PRIVATE
        physica/optimization/lbfgsb/lbfgsb-kernels.h
        physica/optimization/lbfgsb/lbfgsb.cpp
        physica/optimization/lbfgsb/lbfgsb.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/optimization/lbfgsb/lbfgsb.ixx
)
target_include_directories(physica-optimization-lbfgsb PRIVATE "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-optimization-lbfgsb
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
)

# ============================================================================
# Neural
# ============================================================================
add_library(physica-neural STATIC)
target_sources(
        physica-neural
        PRIVATE
        physica/neural/matmul.cpp
        physica/neural/training-state-kernels.h
        physica/neural/training-state.cpp
        physica/neural/training-state.cu
        physica/neural/transformer-kernels.h
        physica/neural/transformer.cpp
        physica/neural/transformer.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/neural/matmul.ixx
        physica/neural/training-state.ixx
        physica/neural/transformer.ixx
)
target_include_directories(physica-neural PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-neural
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        PRIVATE
        CUDA::cublas
        CUDA::cublasLt
)

# ============================================================================
# Simulation / Field
# ============================================================================
add_library(physica-simulation-field STATIC)
target_sources(
        physica-simulation-field
        PRIVATE
        physica/simulation/field/field-kernels.h
        physica/simulation/field/field.cpp
        physica/simulation/field/field.cu
        PUBLIC
        FILE_SET headers TYPE HEADERS
        BASE_DIRS "${PROJECT_SOURCE_DIR}/physica"
        FILES physica/simulation/field/device.cuh
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/simulation/field/field.ixx
)
target_include_directories(physica-simulation-field PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-simulation-field
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-math
)

# ============================================================================
# Fluids / Grid
# ============================================================================
add_library(physica-fluids-grid STATIC)
target_sources(
        physica-fluids-grid
        PUBLIC
        FILE_SET headers TYPE HEADERS
        BASE_DIRS "${PROJECT_SOURCE_DIR}/physica"
        FILES
        physica/fluids/grid/device.cuh
        physica/fluids/grid/interop.h
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/fluids/grid/grid.ixx
        physica/fluids/grid/poisson.ixx
)
target_include_directories(physica-fluids-grid PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-fluids-grid
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-simulation-field
)

# ============================================================================
# Serialization
# ============================================================================
add_library(physica-serialization STATIC)
target_sources(
        physica-serialization
        PRIVATE
        physica/serialization/safetensors.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/serialization/safetensors.ixx
)
target_compile_definitions(physica-serialization PRIVATE PHYSICA_PROJECT_VERSION="${PROJECT_VERSION}")
target_link_libraries(physica-serialization PRIVATE nlohmann_json::nlohmann_json)

# ============================================================================
# Reconstruction / Datasets
# ============================================================================
add_library(physica-reconstruction-dataset STATIC)
target_sources(
        physica-reconstruction-dataset
        PRIVATE
        physica/reconstruction/dataset/ffmpeg.h
        physica/reconstruction/dataset/ffmpeg.cpp
        physica/reconstruction/dataset/nerf_synthetic.cpp
        physica/reconstruction/dataset/pinf.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/reconstruction/dataset/multiview.ixx
        physica/reconstruction/dataset/nerf_synthetic.ixx
        physica/reconstruction/dataset/pinf.ixx
)
target_link_libraries(
        physica-reconstruction-dataset
        PUBLIC
        physica-math
        PRIVATE
        FFmpeg::FFmpeg
        nlohmann_json::nlohmann_json
        physica::stb
)

# ============================================================================
# Fluids / Gas
# ============================================================================
add_library(physica-fluids-gas STATIC)
target_sources(
        physica-fluids-gas
        PRIVATE
        physica/fluids/gas/domain.cpp
        physica/fluids/gas/operators/advection-kernels.h
        physica/fluids/gas/operators/advection-sampling.cuh
        physica/fluids/gas/operators/advection.cpp
        physica/fluids/gas/operators/advection.cu
        physica/fluids/gas/operators/conservation-kernels.h
        physica/fluids/gas/operators/conservation.cpp
        physica/fluids/gas/operators/conservation.cu
        physica/fluids/gas/operators/diffusion-kernels.h
        physica/fluids/gas/operators/diffusion.cpp
        physica/fluids/gas/operators/diffusion.cu
        physica/fluids/gas/operators/force-kernels.h
        physica/fluids/gas/operators/force.cpp
        physica/fluids/gas/operators/force.cu
        physica/fluids/gas/operators/objective-kernels.h
        physica/fluids/gas/operators/objective.cpp
        physica/fluids/gas/operators/objective.cu
        physica/fluids/gas/operators/pointwise-kernels.h
        physica/fluids/gas/operators/pointwise.cpp
        physica/fluids/gas/operators/pointwise.cu
        physica/fluids/gas/operators/projection-kernels.h
        physica/fluids/gas/operators/projection-red-black-gauss-seidel.cu
        physica/fluids/gas/operators/projection.cpp
        physica/fluids/gas/operators/projection.cu
        physica/fluids/gas/solvers/adjoint-control/control-kernels.h
        physica/fluids/gas/solvers/adjoint-control/control.cpp
        physica/fluids/gas/solvers/adjoint-control/control.cu
        physica/fluids/gas/solvers/keyframe-smoke/control-kernels.h
        physica/fluids/gas/solvers/keyframe-smoke/control.cpp
        physica/fluids/gas/solvers/keyframe-smoke/control.cu
        PUBLIC
        FILE_SET headers TYPE HEADERS
        BASE_DIRS "${PROJECT_SOURCE_DIR}/physica"
        FILES
        physica/fluids/gas/device.cuh
        physica/fluids/gas/interop.h
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/fluids/gas/domain.ixx
        physica/fluids/gas/operators/advection.ixx
        physica/fluids/gas/operators/conservation.ixx
        physica/fluids/gas/operators/diffusion.ixx
        physica/fluids/gas/operators/force.ixx
        physica/fluids/gas/operators/objective.ixx
        physica/fluids/gas/operators/pointwise.ixx
        physica/fluids/gas/operators/projection.ixx
        physica/fluids/gas/solvers/adjoint-control/adjoint-control.ixx
        physica/fluids/gas/solvers/adjoint-control/control.ixx
        physica/fluids/gas/solvers/adjoint-control/evaluation.ixx
        physica/fluids/gas/solvers/adjoint-control/optimization.ixx
        physica/fluids/gas/solvers/keyframe-smoke/control.ixx
        physica/fluids/gas/solvers/keyframe-smoke/evaluation.ixx
        physica/fluids/gas/solvers/keyframe-smoke/keyframe-smoke.ixx
        physica/fluids/gas/solvers/keyframe-smoke/multiple-shooting.ixx
        physica/fluids/gas/solvers/keyframe-smoke/optimization.ixx
        physica/fluids/gas/solvers/smoke/smoke.ixx
)
target_include_directories(physica-fluids-gas PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-fluids-gas
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-fluids-grid
        physica-math
        physica-optimization-lbfgsb
)

# ============================================================================
# Fluids / Liquid
# ============================================================================
add_library(physica-fluids-liquid STATIC)
target_sources(
        physica-fluids-liquid
        PRIVATE
        physica/fluids/liquid/operators/density-kernels.h
        physica/fluids/liquid/operators/density.cu
        physica/fluids/liquid/operators/neighborhood-kernels.h
        physica/fluids/liquid/operators/neighborhood.cpp
        physica/fluids/liquid/operators/neighborhood.cu
        physica/fluids/liquid/meshfree.cpp
        physica/fluids/liquid/solvers/pbf/pbf-kernels.h
        physica/fluids/liquid/solvers/pbf/pbf.cpp
        physica/fluids/liquid/solvers/pbf/pbf.cu
        physica/fluids/liquid/solvers/pic/grid-step-kernels.h
        physica/fluids/liquid/solvers/pic/grid-step.cpp
        physica/fluids/liquid/solvers/pic/grid-step.cu
        physica/fluids/liquid/solvers/pic/particle-step-kernels.h
        physica/fluids/liquid/solvers/pic/particle-step.cpp
        physica/fluids/liquid/solvers/pic/particle-step.cu
        physica/fluids/liquid/solvers/pic/projection-kernels.h
        physica/fluids/liquid/solvers/pic/projection.cpp
        physica/fluids/liquid/solvers/pic/projection.cu
        physica/fluids/liquid/solvers/pic/transfer-kernels.h
        physica/fluids/liquid/solvers/pic/transfer.cpp
        physica/fluids/liquid/solvers/pic/transfer.cu
        physica/fluids/liquid/solvers/sph/dynamics-kernels.h
        physica/fluids/liquid/solvers/sph/dynamics-common.cu
        physica/fluids/liquid/solvers/sph/dynamics-dfsph.cu
        physica/fluids/liquid/solvers/sph/dynamics-iisph.cu
        physica/fluids/liquid/solvers/sph/dynamics-pcisph.cu
        physica/fluids/liquid/solvers/sph/dynamics-wcsph.cu
        physica/fluids/liquid/solvers/sph/dynamics.cpp
        PUBLIC
        FILE_SET headers TYPE HEADERS
        BASE_DIRS "${PROJECT_SOURCE_DIR}/physica"
        FILES
        physica/fluids/liquid/device.cuh
        physica/fluids/liquid/interop.h
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/fluids/liquid/operators/density.ixx
        physica/fluids/liquid/operators/neighborhood.ixx
        physica/fluids/liquid/meshfree.ixx
        physica/fluids/liquid/solvers/pbf/pbf.ixx
        physica/fluids/liquid/solvers/pic/grid-step.ixx
        physica/fluids/liquid/solvers/pic/model.ixx
        physica/fluids/liquid/solvers/pic/particle-step.ixx
        physica/fluids/liquid/solvers/pic/pic.ixx
        physica/fluids/liquid/solvers/pic/projection.ixx
        physica/fluids/liquid/solvers/pic/transfer.ixx
        physica/fluids/liquid/solvers/sph/dynamics.ixx
        physica/fluids/liquid/solvers/sph/sph.ixx
)
target_include_directories(physica-fluids-liquid PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-fluids-liquid
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-simulation-field
        physica-fluids-grid
)

# ============================================================================
# Deformables / Cloth
# ============================================================================
add_library(physica-deformables-cloth STATIC)
target_sources(
        physica-deformables-cloth
        PRIVATE
        physica/deformables/cloth/coloring.cpp
        physica/deformables/cloth/topology.cpp
        physica/deformables/cloth/constraints/analytic-collision-kernels.h
        physica/deformables/cloth/constraints/analytic-collision.cpp
        physica/deformables/cloth/constraints/analytic-collision.cu
        physica/deformables/cloth/constraints/fixed-position-kernels.h
        physica/deformables/cloth/constraints/fixed-position.cpp
        physica/deformables/cloth/constraints/fixed-position.cu
        physica/deformables/cloth/constraints/provot-strain-limit-kernels.h
        physica/deformables/cloth/constraints/provot-strain-limit.cpp
        physica/deformables/cloth/constraints/provot-strain-limit.cu
        physica/deformables/cloth/operators/mass-spring-kernels.h
        physica/deformables/cloth/operators/mass-spring.cpp
        physica/deformables/cloth/operators/mass-spring.cu
        physica/deformables/cloth/integrators/forward-euler-kernels.h
        physica/deformables/cloth/integrators/forward-euler.cpp
        physica/deformables/cloth/integrators/forward-euler.cu
        physica/deformables/cloth/integrators/semi-implicit-euler-kernels.h
        physica/deformables/cloth/integrators/semi-implicit-euler.cpp
        physica/deformables/cloth/integrators/semi-implicit-euler.cu
        physica/deformables/cloth/solvers/position-dynamics-kernels.h
        physica/deformables/cloth/solvers/position-dynamics.cu
        physica/deformables/cloth/solvers/velocity-verlet/velocity-verlet-kernels.h
        physica/deformables/cloth/solvers/velocity-verlet/velocity-verlet.cu
        physica/deformables/cloth/solvers/pbd/pbd-kernels.h
        physica/deformables/cloth/solvers/pbd/pbd.cpp
        physica/deformables/cloth/solvers/pbd/pbd.cu
        physica/deformables/cloth/solvers/strain-based-dynamics/strain-based-dynamics-kernels.h
        physica/deformables/cloth/solvers/strain-based-dynamics/strain-based-dynamics.cpp
        physica/deformables/cloth/solvers/strain-based-dynamics/strain-based-dynamics.cu
        physica/deformables/cloth/solvers/xpbd/xpbd-kernels.h
        physica/deformables/cloth/solvers/xpbd/xpbd.cpp
        physica/deformables/cloth/solvers/xpbd/xpbd.cu
        physica/deformables/cloth/solvers/fast-projection/fast-projection-kernels.h
        physica/deformables/cloth/solvers/fast-projection/fast-projection.cpp
        physica/deformables/cloth/solvers/fast-projection/fast-projection.cu
        physica/deformables/cloth/solvers/block-pcg/block-pcg-kernels.h
        physica/deformables/cloth/solvers/block-pcg/block-pcg.cpp
        physica/deformables/cloth/solvers/block-pcg/block-pcg.cu
        physica/deformables/cloth/solvers/fast-mass-spring/fast-mass-spring-kernels.h
        physica/deformables/cloth/solvers/fast-mass-spring/fast-mass-spring.cpp
        physica/deformables/cloth/solvers/fast-mass-spring/fast-mass-spring.cu
        physica/deformables/cloth/solvers/baraff-witkin/baraff-witkin-kernels.h
        physica/deformables/cloth/solvers/baraff-witkin/second-order.cuh
        physica/deformables/cloth/solvers/baraff-witkin/baraff-witkin.cpp
        physica/deformables/cloth/solvers/baraff-witkin/baraff-witkin.cu
        physica/deformables/cloth/solvers/projective-dynamics/projective-dynamics-kernels.h
        physica/deformables/cloth/solvers/projective-dynamics/projective-dynamics.cpp
        physica/deformables/cloth/solvers/projective-dynamics/projective-dynamics.cu
        physica/deformables/cloth/solvers/choi-ko/choi-ko-kernels.h
        physica/deformables/cloth/solvers/choi-ko/choi-ko.cpp
        physica/deformables/cloth/solvers/choi-ko/choi-ko.cu
        physica/deformables/cloth/solvers/stvk-fem/stvk-fem-kernels.h
        physica/deformables/cloth/solvers/stvk-fem/stvk-fem.cpp
        physica/deformables/cloth/solvers/stvk-fem/stvk-fem.cu
        physica/deformables/cloth/solvers/corotated-fem/corotated-fem-kernels.h
        physica/deformables/cloth/solvers/corotated-fem/corotated-fem.cpp
        physica/deformables/cloth/solvers/corotated-fem/corotated-fem.cu
        physica/deformables/cloth/solvers/surface-neo-hookean/surface-neo-hookean-kernels.h
        physica/deformables/cloth/solvers/surface-neo-hookean/surface-neo-hookean.cpp
        physica/deformables/cloth/solvers/surface-neo-hookean/surface-neo-hookean.cu
        physica/deformables/cloth/solvers/discrete-shells/discrete-shells-kernels.h
        physica/deformables/cloth/solvers/discrete-shells/second-order.cuh
        physica/deformables/cloth/solvers/discrete-shells/discrete-shells.cpp
        physica/deformables/cloth/solvers/discrete-shells/discrete-shells.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/deformables/cloth/coloring.ixx
        physica/deformables/cloth/topology.ixx
        physica/deformables/cloth/model.ixx
        physica/deformables/cloth/state.ixx
        physica/deformables/cloth/forward.ixx
        physica/deformables/cloth/constraints/analytic-collision.ixx
        physica/deformables/cloth/constraints/fixed-position.ixx
        physica/deformables/cloth/constraints/provot-strain-limit.ixx
        physica/deformables/cloth/operators/mass-spring.ixx
        physica/deformables/cloth/integrators/forward-euler.ixx
        physica/deformables/cloth/integrators/semi-implicit-euler.ixx
        physica/deformables/cloth/solvers/explicit-dynamics/explicit-dynamics.ixx
        physica/deformables/cloth/solvers/velocity-verlet/velocity-verlet.ixx
        physica/deformables/cloth/solvers/pbd/pbd.ixx
        physica/deformables/cloth/solvers/strain-based-dynamics/strain-based-dynamics.ixx
        physica/deformables/cloth/solvers/xpbd/xpbd.ixx
        physica/deformables/cloth/solvers/fast-projection/fast-projection.ixx
        physica/deformables/cloth/solvers/block-pcg/block-pcg.ixx
        physica/deformables/cloth/solvers/fast-mass-spring/fast-mass-spring.ixx
        physica/deformables/cloth/solvers/baraff-witkin/baraff-witkin.ixx
        physica/deformables/cloth/solvers/projective-dynamics/projective-dynamics.ixx
        physica/deformables/cloth/solvers/choi-ko/choi-ko.ixx
        physica/deformables/cloth/solvers/stvk-fem/stvk-fem.ixx
        physica/deformables/cloth/solvers/corotated-fem/corotated-fem.ixx
        physica/deformables/cloth/solvers/surface-neo-hookean/surface-neo-hookean.ixx
        physica/deformables/cloth/solvers/discrete-shells/discrete-shells.ixx
)
target_include_directories(physica-deformables-cloth PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-deformables-cloth
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-simulation-field
        PRIVATE
        cudss
        CUDA::cublas
)

# ============================================================================
# Reconstruction / Instant-NGP
# ============================================================================
add_library(physica-reconstruction-instant-ngp STATIC)
target_sources(
        physica-reconstruction-instant-ngp
        PRIVATE
        physica/reconstruction/instant-ngp/instant-ngp.cpp
        physica/reconstruction/instant-ngp/network/kernels.h
        physica/reconstruction/instant-ngp/network/network.cpp
        physica/reconstruction/instant-ngp/network/network.cu
        physica/reconstruction/instant-ngp/rendering/kernels.h
        physica/reconstruction/instant-ngp/rendering/rendering.cpp
        physica/reconstruction/instant-ngp/rendering/rendering.cu
        physica/reconstruction/instant-ngp/sampling/kernels.h
        physica/reconstruction/instant-ngp/sampling/sampling.cpp
        physica/reconstruction/instant-ngp/sampling/sampling.cu
        physica/reconstruction/instant-ngp/scene/scene.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/reconstruction/instant-ngp/instant-ngp.ixx
        physica/reconstruction/instant-ngp/network/network.ixx
        physica/reconstruction/instant-ngp/rendering/rendering.ixx
        physica/reconstruction/instant-ngp/sampling/sampling.ixx
        physica/reconstruction/instant-ngp/scene/scene.ixx
)
target_include_directories(physica-reconstruction-instant-ngp PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-reconstruction-instant-ngp
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-reconstruction-dataset
        physica-math
        PRIVATE
        CUDA::cublas
        CUDA::cublasLt
        physica-serialization
)

# ============================================================================
# Reconstruction / PINFS
# ============================================================================
add_library(physica-reconstruction-pinfs STATIC)
target_sources(
        physica-reconstruction-pinfs
        PRIVATE
        physica/reconstruction/pinfs/pinfs.cpp
        physica/reconstruction/pinfs/field/field.cpp
        physica/reconstruction/pinfs/field/field.cu
        physica/reconstruction/pinfs/field/kernels.h
        physica/reconstruction/pinfs/network/network.cpp
        physica/reconstruction/pinfs/network/network.cu
        physica/reconstruction/pinfs/network/kernels.h
        physica/reconstruction/pinfs/perceptual/perceptual.cpp
        physica/reconstruction/pinfs/perceptual/perceptual.cu
        physica/reconstruction/pinfs/perceptual/kernels.h
        physica/reconstruction/pinfs/physics/physics.cpp
        physica/reconstruction/pinfs/physics/physics.cu
        physica/reconstruction/pinfs/physics/kernels.h
        physica/reconstruction/pinfs/rendering/rendering.cpp
        physica/reconstruction/pinfs/rendering/rendering.cu
        physica/reconstruction/pinfs/rendering/kernels.h
        physica/reconstruction/pinfs/sampling/sampling.cpp
        physica/reconstruction/pinfs/sampling/sampling.cu
        physica/reconstruction/pinfs/sampling/kernels.h
        physica/reconstruction/pinfs/scene/scene.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/reconstruction/pinfs/pinfs.ixx
        physica/reconstruction/pinfs/field/field.ixx
        physica/reconstruction/pinfs/network/network.ixx
        physica/reconstruction/pinfs/perceptual/perceptual.ixx
        physica/reconstruction/pinfs/physics/physics.ixx
        physica/reconstruction/pinfs/rendering/rendering.ixx
        physica/reconstruction/pinfs/sampling/sampling.ixx
        physica/reconstruction/pinfs/scene/scene.ixx
)
target_include_directories(physica-reconstruction-pinfs PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-reconstruction-pinfs
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-reconstruction-dataset
        physica-math
        PRIVATE
        CUDA::cublas
        physica-serialization
)

# ============================================================================
# Generative / Flow Matching
# ============================================================================
add_library(physica-generative-flow-matching STATIC)
target_sources(
        physica-generative-flow-matching
        PRIVATE
        physica/generative/dataset/cifar10.cpp
        physica/generative/flow-matching/kernels.h
        physica/generative/flow-matching/kernels.cu
        physica/generative/flow-matching/model.cpp
        physica/generative/flow-matching/sampling.cpp
        physica/generative/flow-matching/flow-matching.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/generative/dataset/cifar10.ixx
        physica/generative/flow-matching/model.ixx
        physica/generative/flow-matching/sampling.ixx
        physica/generative/flow-matching/flow-matching.ixx
)
target_include_directories(physica-generative-flow-matching PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-generative-flow-matching
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-neural
        PRIVATE
        CUDA::curand
        physica-serialization
)

if (BUILD_TESTING)
    add_executable(
            physica-flow-matching-regression
            physica/generative/flow-matching/regression.cpp
    )
    target_link_libraries(
            physica-flow-matching-regression
            PRIVATE
            physica-generative-flow-matching
            physica-neural
            physica-serialization
    )
    add_test(NAME physica-flow-matching-regression COMMAND physica-flow-matching-regression)
endif ()

# ============================================================================
# Examples
# ============================================================================
if (PHYSICA_EXAMPLES)
    add_executable(
            physica-example-flow-matching-cli
            examples/generative/flow-matching/main.cpp
    )
    target_link_libraries(
            physica-example-flow-matching-cli
            PRIVATE
            physica-generative-flow-matching
            physica::stb
    )

    add_executable(
            physica-example-instant-ngp-cli
            examples/reconstruction/instant-ngp/main.cpp
    )
    target_link_libraries(
            physica-example-instant-ngp-cli
            PRIVATE
            physica-reconstruction-dataset
            physica-reconstruction-instant-ngp
    )

    add_executable(
            physica-example-pinfs-cli
            examples/reconstruction/pinfs/main.cpp
    )
    target_link_libraries(
            physica-example-pinfs-cli
            PRIVATE
            physica-reconstruction-dataset
            physica-reconstruction-pinfs
            physica::stb
    )
    if (WIN32)
        add_custom_command(
                TARGET physica-example-pinfs-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-pinfs-cli> $<TARGET_FILE_DIR:physica-example-pinfs-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-smoke STATIC)
    target_sources(
            physica-example-smoke
            PRIVATE
            examples/fluids/gas/smoke/simulation.cu
            examples/fluids/gas/smoke/simulation_kernels.h
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/fluids/gas/smoke/simulation.ixx
    )
    target_link_libraries(physica-example-smoke PUBLIC physica-fluids-gas)
    add_executable(
            physica-example-smoke-cli
            examples/fluids/gas/smoke/main.cpp
    )
    target_link_libraries(physica-example-smoke-cli PRIVATE physica-example-smoke)

    add_executable(
            physica-example-keyframe-smoke-cli
            examples/fluids/gas/keyframe-smoke/experiment.cpp
            examples/fluids/gas/keyframe-smoke/main.cpp
    )
    target_sources(
            physica-example-keyframe-smoke-cli
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/fluids/gas/keyframe-smoke/experiment.ixx
    )
    target_link_libraries(
            physica-example-keyframe-smoke-cli
            PRIVATE
            physica-fluids-gas
            physica::stb
    )

    add_executable(
            physica-example-adjoint-control-cli
            examples/fluids/gas/adjoint-control/experiment.cpp
            examples/fluids/gas/adjoint-control/main.cpp
    )
    target_sources(
            physica-example-adjoint-control-cli
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/fluids/gas/adjoint-control/experiment.ixx
    )
    target_link_libraries(
            physica-example-adjoint-control-cli
            PRIVATE
            physica-fluids-gas
            physica::stb
    )

    add_library(physica-example-pbf-dam-break STATIC)
    target_sources(
            physica-example-pbf-dam-break
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/fluids/liquid/pbf-dam-break/simulation.ixx
    )
    target_link_libraries(physica-example-pbf-dam-break PUBLIC physica-fluids-liquid)
    add_executable(
            physica-example-pbf-dam-break-cli
            examples/fluids/liquid/pbf-dam-break/main.cpp
    )
    target_link_libraries(physica-example-pbf-dam-break-cli PRIVATE physica-example-pbf-dam-break)

    add_library(physica-example-flip-apic-dam-break STATIC)
    target_sources(
            physica-example-flip-apic-dam-break
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/fluids/liquid/flip-apic-dam-break/simulation.ixx
    )
    target_link_libraries(physica-example-flip-apic-dam-break PUBLIC physica-fluids-liquid)
    add_executable(
            physica-example-flip-apic-dam-break-cli
            examples/fluids/liquid/flip-apic-dam-break/main.cpp
    )
    target_link_libraries(physica-example-flip-apic-dam-break-cli PRIVATE physica-example-flip-apic-dam-break)

    add_library(physica-example-cloth-support STATIC)
    target_sources(
            physica-example-cloth-support
            PRIVATE
            examples/deformables/cloth/support/scene.cpp
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/support/scene.ixx
    )
    target_link_libraries(physica-example-cloth-support PUBLIC physica-deformables-cloth)

    add_library(physica-example-cloth-forward-euler STATIC)
    target_sources(
            physica-example-cloth-forward-euler
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/forward-euler/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-forward-euler PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-forward-euler-cli
            examples/deformables/cloth/forward-euler/main.cpp
    )
    target_link_libraries(physica-example-cloth-forward-euler-cli PRIVATE physica-example-cloth-forward-euler)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-forward-euler-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-forward-euler-cli> $<TARGET_FILE_DIR:physica-example-cloth-forward-euler-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-semi-implicit-euler STATIC)
    target_sources(
            physica-example-cloth-semi-implicit-euler
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/semi-implicit-euler/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-semi-implicit-euler PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-semi-implicit-euler-cli
            examples/deformables/cloth/semi-implicit-euler/main.cpp
    )
    target_link_libraries(physica-example-cloth-semi-implicit-euler-cli PRIVATE physica-example-cloth-semi-implicit-euler)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-semi-implicit-euler-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-semi-implicit-euler-cli> $<TARGET_FILE_DIR:physica-example-cloth-semi-implicit-euler-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-velocity-verlet STATIC)
    target_sources(
            physica-example-cloth-velocity-verlet
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/velocity-verlet/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-velocity-verlet PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-velocity-verlet-cli
            examples/deformables/cloth/velocity-verlet/main.cpp
    )
    target_link_libraries(physica-example-cloth-velocity-verlet-cli PRIVATE physica-example-cloth-velocity-verlet)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-velocity-verlet-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-velocity-verlet-cli> $<TARGET_FILE_DIR:physica-example-cloth-velocity-verlet-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-provot-strain-limiting STATIC)
    target_sources(
            physica-example-cloth-provot-strain-limiting
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/provot-strain-limiting/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-provot-strain-limiting PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-provot-strain-limiting-cli
            examples/deformables/cloth/provot-strain-limiting/main.cpp
    )
    target_link_libraries(physica-example-cloth-provot-strain-limiting-cli PRIVATE physica-example-cloth-provot-strain-limiting)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-provot-strain-limiting-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-provot-strain-limiting-cli> $<TARGET_FILE_DIR:physica-example-cloth-provot-strain-limiting-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-analytic-collision STATIC)
    target_sources(
            physica-example-cloth-analytic-collision
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/analytic-collision/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-analytic-collision PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-analytic-collision-cli
            examples/deformables/cloth/analytic-collision/main.cpp
    )
    target_link_libraries(physica-example-cloth-analytic-collision-cli PRIVATE physica-example-cloth-analytic-collision)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-analytic-collision-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-analytic-collision-cli> $<TARGET_FILE_DIR:physica-example-cloth-analytic-collision-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-pbd STATIC)
    target_sources(
            physica-example-cloth-pbd
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/pbd/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-pbd PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-pbd-cli
            examples/deformables/cloth/pbd/main.cpp
    )
    target_link_libraries(physica-example-cloth-pbd-cli PRIVATE physica-example-cloth-pbd)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-pbd-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-pbd-cli> $<TARGET_FILE_DIR:physica-example-cloth-pbd-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-strain-based-dynamics STATIC)
    target_sources(
            physica-example-cloth-strain-based-dynamics
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/strain-based-dynamics/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-strain-based-dynamics PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-strain-based-dynamics-cli
            examples/deformables/cloth/strain-based-dynamics/main.cpp
    )
    target_link_libraries(physica-example-cloth-strain-based-dynamics-cli PRIVATE physica-example-cloth-strain-based-dynamics)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-strain-based-dynamics-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-strain-based-dynamics-cli> $<TARGET_FILE_DIR:physica-example-cloth-strain-based-dynamics-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-xpbd STATIC)
    target_sources(
            physica-example-cloth-xpbd
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/xpbd/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-xpbd PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-xpbd-cli
            examples/deformables/cloth/xpbd/main.cpp
    )
    target_link_libraries(physica-example-cloth-xpbd-cli PRIVATE physica-example-cloth-xpbd)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-xpbd-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-xpbd-cli> $<TARGET_FILE_DIR:physica-example-cloth-xpbd-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-fast-projection STATIC)
    target_sources(
            physica-example-cloth-fast-projection
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/fast-projection/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-fast-projection PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-fast-projection-cli
            examples/deformables/cloth/fast-projection/main.cpp
    )
    target_link_libraries(physica-example-cloth-fast-projection-cli PRIVATE physica-example-cloth-fast-projection)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-fast-projection-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-fast-projection-cli> $<TARGET_FILE_DIR:physica-example-cloth-fast-projection-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-fast-mass-spring STATIC)
    target_sources(
            physica-example-cloth-fast-mass-spring
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/fast-mass-spring/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-fast-mass-spring PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-fast-mass-spring-cli
            examples/deformables/cloth/fast-mass-spring/main.cpp
    )
    target_link_libraries(physica-example-cloth-fast-mass-spring-cli PRIVATE physica-example-cloth-fast-mass-spring)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-fast-mass-spring-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-fast-mass-spring-cli> $<TARGET_FILE_DIR:physica-example-cloth-fast-mass-spring-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-baraff-witkin STATIC)
    target_sources(
            physica-example-cloth-baraff-witkin
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/baraff-witkin/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-baraff-witkin PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-baraff-witkin-cli
            examples/deformables/cloth/baraff-witkin/main.cpp
    )
    target_link_libraries(physica-example-cloth-baraff-witkin-cli PRIVATE physica-example-cloth-baraff-witkin)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-baraff-witkin-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-baraff-witkin-cli> $<TARGET_FILE_DIR:physica-example-cloth-baraff-witkin-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-projective-dynamics STATIC)
    target_sources(
            physica-example-cloth-projective-dynamics
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/projective-dynamics/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-projective-dynamics PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-projective-dynamics-cli
            examples/deformables/cloth/projective-dynamics/main.cpp
    )
    target_link_libraries(physica-example-cloth-projective-dynamics-cli PRIVATE physica-example-cloth-projective-dynamics)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-projective-dynamics-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-projective-dynamics-cli> $<TARGET_FILE_DIR:physica-example-cloth-projective-dynamics-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-choi-ko STATIC)
    target_sources(
            physica-example-cloth-choi-ko
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/choi-ko/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-choi-ko PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-choi-ko-cli
            examples/deformables/cloth/choi-ko/main.cpp
    )
    target_link_libraries(physica-example-cloth-choi-ko-cli PRIVATE physica-example-cloth-choi-ko)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-choi-ko-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-choi-ko-cli> $<TARGET_FILE_DIR:physica-example-cloth-choi-ko-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-stvk-fem STATIC)
    target_sources(
            physica-example-cloth-stvk-fem
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/stvk-fem/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-stvk-fem PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-stvk-fem-cli
            examples/deformables/cloth/stvk-fem/main.cpp
    )
    target_link_libraries(physica-example-cloth-stvk-fem-cli PRIVATE physica-example-cloth-stvk-fem)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-stvk-fem-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-stvk-fem-cli> $<TARGET_FILE_DIR:physica-example-cloth-stvk-fem-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-corotated-fem STATIC)
    target_sources(
            physica-example-cloth-corotated-fem
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/corotated-fem/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-corotated-fem PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-corotated-fem-cli
            examples/deformables/cloth/corotated-fem/main.cpp
    )
    target_link_libraries(physica-example-cloth-corotated-fem-cli PRIVATE physica-example-cloth-corotated-fem)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-corotated-fem-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-corotated-fem-cli> $<TARGET_FILE_DIR:physica-example-cloth-corotated-fem-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-surface-neo-hookean STATIC)
    target_sources(
            physica-example-cloth-surface-neo-hookean
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/surface-neo-hookean/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-surface-neo-hookean PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-surface-neo-hookean-cli
            examples/deformables/cloth/surface-neo-hookean/main.cpp
    )
    target_link_libraries(physica-example-cloth-surface-neo-hookean-cli PRIVATE physica-example-cloth-surface-neo-hookean)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-surface-neo-hookean-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-surface-neo-hookean-cli> $<TARGET_FILE_DIR:physica-example-cloth-surface-neo-hookean-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    add_library(physica-example-cloth-discrete-shells STATIC)
    target_sources(
            physica-example-cloth-discrete-shells
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/discrete-shells/simulation.ixx
    )
    target_link_libraries(physica-example-cloth-discrete-shells PUBLIC physica-example-cloth-support)
    add_executable(
            physica-example-cloth-discrete-shells-cli
            examples/deformables/cloth/discrete-shells/main.cpp
    )
    target_link_libraries(physica-example-cloth-discrete-shells-cli PRIVATE physica-example-cloth-discrete-shells)
    if (WIN32)
        add_custom_command(
                TARGET physica-example-cloth-discrete-shells-cli
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:physica-example-cloth-discrete-shells-cli> $<TARGET_FILE_DIR:physica-example-cloth-discrete-shells-cli>
                COMMAND_EXPAND_LISTS
        )
    endif ()

    if (PHYSICA_BUILD_SPECTRA)
        physica_require_dependency(spectra_sdk)

        spectra_add_provider(
                physica-example-cloth-flag-spectra
                MODULE physica.example.deformables.cloth.flag.module
                TYPE physica::examples::cloth_flag::Module
        )
        target_sources(
                physica-example-cloth-flag-spectra
                PRIVATE
                examples/deformables/cloth/flag/simulation_kernels.h
                examples/deformables/cloth/flag/simulation.cu
                examples/deformables/cloth/flag/module_kernels.h
                examples/deformables/cloth/flag/module_kernels.cu
                examples/deformables/cloth/flag/module.cpp
                PRIVATE
                FILE_SET provider_module TYPE CXX_MODULES FILES
                examples/deformables/cloth/flag/module.ixx
        )
        target_link_libraries(physica-example-cloth-flag-spectra PRIVATE physica-deformables-cloth)
        set(physica_cloth_flag_bundle_directory "${PROJECT_BINARY_DIR}/spectra/cloth-flag")
        file(MAKE_DIRECTORY "${physica_cloth_flag_bundle_directory}/textures")
        configure_file(
                examples/deformables/cloth/flag/scene.usda
                "${physica_cloth_flag_bundle_directory}/scene.usda"
                COPYONLY
        )
        configure_file(
                examples/deformables/cloth/flag/scene.physica.usda
                "${physica_cloth_flag_bundle_directory}/scene.physica.usda"
                COPYONLY
        )
        configure_file(
                examples/deformables/cloth/flag/textures/color_0C0C0C.exr
                "${physica_cloth_flag_bundle_directory}/textures/color_0C0C0C.exr"
                COPYONLY
        )
        set_target_properties(
                physica-example-cloth-flag-spectra
                PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${physica_cloth_flag_bundle_directory}>"
        )

        spectra_add_provider(
                physica-example-instant-ngp-spectra
                MODULE physica.example.reconstruction.instant_ngp.module
                TYPE physica::examples::instant_ngp::Module
        )
        target_sources(
                physica-example-instant-ngp-spectra
                PRIVATE
                examples/reconstruction/instant-ngp/spectra/module.cpp
                PRIVATE
                FILE_SET provider_module TYPE CXX_MODULES FILES
                examples/reconstruction/instant-ngp/spectra/module.ixx
        )
        target_link_libraries(
                physica-example-instant-ngp-spectra
                PRIVATE
                physica-reconstruction-dataset
                physica-reconstruction-instant-ngp
        )
        set(physica_instant_ngp_bundle_directory "${PROJECT_BINARY_DIR}/spectra/instant-ngp")
        file(MAKE_DIRECTORY "${physica_instant_ngp_bundle_directory}")
        configure_file(
                examples/reconstruction/instant-ngp/spectra/scene.usda
                "${physica_instant_ngp_bundle_directory}/scene.usda"
                COPYONLY
        )
        configure_file(
                examples/reconstruction/instant-ngp/spectra/scene.physica.usda
                "${physica_instant_ngp_bundle_directory}/scene.physica.usda"
                COPYONLY
        )
        set_target_properties(
                physica-example-instant-ngp-spectra
                PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${physica_instant_ngp_bundle_directory}>"
        )

        spectra_add_provider(
                physica-example-smoke-spectra
                MODULE physica.example.fluids.gas.smoke.module
                TYPE physica::examples::smoke::Module
        )
        target_sources(
                physica-example-smoke-spectra
                PRIVATE
                examples/fluids/gas/smoke/spectra/module.cpp
                PRIVATE
                FILE_SET provider_module TYPE CXX_MODULES FILES
                examples/fluids/gas/smoke/spectra/module.ixx
        )
        target_link_libraries(physica-example-smoke-spectra PRIVATE physica-example-smoke)
        set(physica_smoke_bundle_directory "${PROJECT_BINARY_DIR}/spectra/smoke")
        file(MAKE_DIRECTORY "${physica_smoke_bundle_directory}")
        configure_file(
                examples/fluids/gas/smoke/spectra/scene.usda
                "${physica_smoke_bundle_directory}/scene.usda"
                COPYONLY
        )
        set_target_properties(
                physica-example-smoke-spectra
                PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${physica_smoke_bundle_directory}>"
        )

        spectra_add_provider(
                physica-example-pbf-dam-break-spectra
                MODULE physica.example.fluids.liquid.pbf_dam_break.module
                TYPE physica::examples::pbf_dam_break::Module
        )
        target_sources(
                physica-example-pbf-dam-break-spectra
                PRIVATE
                examples/fluids/liquid/pbf-dam-break/spectra/kernels.cu
                examples/fluids/liquid/pbf-dam-break/spectra/kernels.h
                examples/fluids/liquid/pbf-dam-break/spectra/module.cpp
                PRIVATE
                FILE_SET provider_module TYPE CXX_MODULES FILES
                examples/fluids/liquid/pbf-dam-break/spectra/module.ixx
        )
        target_link_libraries(physica-example-pbf-dam-break-spectra PRIVATE physica-example-pbf-dam-break)
        set(physica_pbf_bundle_directory "${PROJECT_BINARY_DIR}/spectra/pbf-dam-break")
        file(MAKE_DIRECTORY "${physica_pbf_bundle_directory}")
        configure_file(
                examples/fluids/liquid/pbf-dam-break/spectra/scene.usda
                "${physica_pbf_bundle_directory}/scene.usda"
                COPYONLY
        )
        set_target_properties(
                physica-example-pbf-dam-break-spectra
                PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${physica_pbf_bundle_directory}>"
        )

        spectra_add_provider(
                physica-example-flip-apic-dam-break-spectra
                MODULE physica.example.fluids.liquid.flip_apic_dam_break.module
                TYPE physica::examples::flip_apic_dam_break::Module
        )
        target_sources(
                physica-example-flip-apic-dam-break-spectra
                PRIVATE
                examples/fluids/liquid/flip-apic-dam-break/spectra/kernels.cu
                examples/fluids/liquid/flip-apic-dam-break/spectra/kernels.h
                examples/fluids/liquid/flip-apic-dam-break/spectra/module.cpp
                PRIVATE
                FILE_SET provider_module TYPE CXX_MODULES FILES
                examples/fluids/liquid/flip-apic-dam-break/spectra/module.ixx
        )
        target_link_libraries(physica-example-flip-apic-dam-break-spectra PRIVATE physica-example-flip-apic-dam-break)
        set(physica_flip_apic_bundle_directory "${PROJECT_BINARY_DIR}/spectra/flip-apic-dam-break")
        file(MAKE_DIRECTORY "${physica_flip_apic_bundle_directory}")
        configure_file(
                examples/fluids/liquid/flip-apic-dam-break/spectra/scene.usda
                "${physica_flip_apic_bundle_directory}/scene.usda"
                COPYONLY
        )
        set_target_properties(
                physica-example-flip-apic-dam-break-spectra
                PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${physica_flip_apic_bundle_directory}>"
        )

    endif ()
endif ()

# MSVC interprets a bare @cmake_cxx_std.lib as a response file.
if (MSVC)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/cxx-modules")
endif ()
