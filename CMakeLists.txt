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
        physica/deformables/cloth/model.cpp
        physica/deformables/cloth/operators/fixed-position-kernels.h
        physica/deformables/cloth/operators/fixed-position.cpp
        physica/deformables/cloth/operators/fixed-position.cu
        physica/deformables/cloth/operators/mass-spring-kernels.h
        physica/deformables/cloth/operators/mass-spring.cpp
        physica/deformables/cloth/operators/mass-spring.cu
        physica/deformables/cloth/operators/semi-implicit-euler-kernels.h
        physica/deformables/cloth/operators/semi-implicit-euler.cpp
        physica/deformables/cloth/operators/semi-implicit-euler.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/deformables/cloth/model.ixx
        physica/deformables/cloth/operators/fixed-position.ixx
        physica/deformables/cloth/operators/mass-spring.ixx
        physica/deformables/cloth/operators/semi-implicit-euler.ixx
        physica/deformables/cloth/solvers/explicit-dynamics/explicit-dynamics.ixx
)
target_include_directories(physica-deformables-cloth PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-deformables-cloth
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-simulation-field
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

    add_library(physica-example-cloth STATIC)
    target_sources(
            physica-example-cloth
            PRIVATE
            examples/deformables/cloth/simulation.cu
            examples/deformables/cloth/simulation_kernels.h
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/deformables/cloth/simulation.ixx
    )
    target_link_libraries(physica-example-cloth PUBLIC physica-deformables-cloth)
    add_executable(
            physica-example-cloth-cli
            examples/deformables/cloth/main.cpp
    )
    target_link_libraries(physica-example-cloth-cli PRIVATE physica-example-cloth)

    if (PHYSICA_BUILD_SPECTRA)
        physica_require_dependency(spectra_sdk)

        spectra_add_provider(
                physica-example-instant-ngp-spectra
                MODULE physica.example.reconstruction.instant_ngp.spectra
                TYPE physica::examples::instant_ngp::Provider
        )
        target_sources(
                physica-example-instant-ngp-spectra
                PRIVATE
                FILE_SET provider_module TYPE CXX_MODULES FILES
                examples/reconstruction/instant-ngp/spectra/provider.ixx
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
                examples/reconstruction/instant-ngp/spectra/scene.spectra
                "${physica_instant_ngp_bundle_directory}/scene.spectra"
                COPYONLY
        )
        set_target_properties(
                physica-example-instant-ngp-spectra
                PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${physica_instant_ngp_bundle_directory}>"
        )

        spectra_add_provider(
                physica-example-smoke-spectra
                MODULE physica.example.fluids.gas.smoke.spectra
                TYPE physica::examples::smoke::Provider
        )
        target_sources(
                physica-example-smoke-spectra
                PRIVATE
                FILE_SET provider_module TYPE CXX_MODULES FILES
                examples/fluids/gas/smoke/spectra/provider.ixx
        )
        target_link_libraries(physica-example-smoke-spectra PRIVATE physica-example-smoke)
        set(physica_smoke_bundle_directory "${PROJECT_BINARY_DIR}/spectra/smoke")
        file(MAKE_DIRECTORY "${physica_smoke_bundle_directory}")
        configure_file(
                examples/fluids/gas/smoke/spectra/scene.spectra
                "${physica_smoke_bundle_directory}/scene.spectra"
                COPYONLY
        )
        set_target_properties(
                physica-example-smoke-spectra
                PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${physica_smoke_bundle_directory}>"
        )

        spectra_add_provider(
                physica-example-pbf-dam-break-spectra
                MODULE physica.example.fluids.liquid.pbf_dam_break.spectra
                TYPE physica::examples::pbf_dam_break::Provider
        )
        target_sources(
                physica-example-pbf-dam-break-spectra
                PRIVATE
                examples/fluids/liquid/pbf-dam-break/spectra/kernels.cu
                examples/fluids/liquid/pbf-dam-break/spectra/kernels.h
                PRIVATE
                FILE_SET provider_module TYPE CXX_MODULES FILES
                examples/fluids/liquid/pbf-dam-break/spectra/provider.ixx
        )
        target_link_libraries(physica-example-pbf-dam-break-spectra PRIVATE physica-example-pbf-dam-break)
        set(physica_pbf_bundle_directory "${PROJECT_BINARY_DIR}/spectra/pbf-dam-break")
        file(MAKE_DIRECTORY "${physica_pbf_bundle_directory}")
        configure_file(
                examples/fluids/liquid/pbf-dam-break/spectra/scene.spectra
                "${physica_pbf_bundle_directory}/scene.spectra"
                COPYONLY
        )
        set_target_properties(
                physica-example-pbf-dam-break-spectra
                PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${physica_pbf_bundle_directory}>"
        )

        spectra_add_provider(
                physica-example-flip-apic-dam-break-spectra
                MODULE physica.example.fluids.liquid.flip_apic_dam_break.spectra
                TYPE physica::examples::flip_apic_dam_break::Provider
        )
        target_sources(
                physica-example-flip-apic-dam-break-spectra
                PRIVATE
                examples/fluids/liquid/flip-apic-dam-break/spectra/kernels.cu
                examples/fluids/liquid/flip-apic-dam-break/spectra/kernels.h
                PRIVATE
                FILE_SET provider_module TYPE CXX_MODULES FILES
                examples/fluids/liquid/flip-apic-dam-break/spectra/provider.ixx
        )
        target_link_libraries(physica-example-flip-apic-dam-break-spectra PRIVATE physica-example-flip-apic-dam-break)
        set(physica_flip_apic_bundle_directory "${PROJECT_BINARY_DIR}/spectra/flip-apic-dam-break")
        file(MAKE_DIRECTORY "${physica_flip_apic_bundle_directory}")
        configure_file(
                examples/fluids/liquid/flip-apic-dam-break/spectra/scene.spectra
                "${physica_flip_apic_bundle_directory}/scene.spectra"
                COPYONLY
        )
        set_target_properties(
                physica-example-flip-apic-dam-break-spectra
                PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${physica_flip_apic_bundle_directory}>"
        )

        spectra_add_provider(
                physica-example-cloth-spectra
                MODULE physica.example.deformables.cloth.spectra
                TYPE physica::examples::cloth::Provider
        )
        target_sources(
                physica-example-cloth-spectra
                PRIVATE
                examples/deformables/cloth/spectra/kernels.cu
                examples/deformables/cloth/spectra/kernels.h
                PRIVATE
                FILE_SET provider_module TYPE CXX_MODULES FILES
                examples/deformables/cloth/spectra/provider.ixx
        )
        target_link_libraries(physica-example-cloth-spectra PRIVATE physica-example-cloth)
        set(physica_cloth_bundle_directory "${PROJECT_BINARY_DIR}/spectra/cloth")
        file(MAKE_DIRECTORY "${physica_cloth_bundle_directory}/assets")
        configure_file(
                examples/deformables/cloth/spectra/scene.spectra
                "${physica_cloth_bundle_directory}/scene.spectra"
                COPYONLY
        )
        configure_file(
                examples/deformables/cloth/spectra/assets/flag.ply
                "${physica_cloth_bundle_directory}/assets/flag.ply"
                COPYONLY
        )
        set_target_properties(
                physica-example-cloth-spectra
                PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${physica_cloth_bundle_directory}>"
        )
    endif ()
endif ()

# MSVC interprets a bare @cmake_cxx_std.lib as a response file.
if (MSVC)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/cxx-modules")
endif ()
