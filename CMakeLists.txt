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

# ============================================================================
# Dependencies
# ============================================================================
include(cmake/Dependencies.cmake)
physica_require_dependency(cuda)
physica_require_dependency(cccl)
physica_require_dependency(nlohmann_json)
physica_require_dependency(stb)
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
        BASE_DIRS physica/math/include
        FILES physica/math/include/physica/math.h
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/math/math.ixx
)
target_link_libraries(physica-math PUBLIC CCCL::libcudacxx)

# ============================================================================
# Field
# ============================================================================
add_library(physica-field STATIC)
target_sources(
        physica-field
        PRIVATE
        physica/field/field-kernels.h
        physica/field/field.cpp
        physica/field/field.cu
        PUBLIC
        FILE_SET headers TYPE HEADERS
        BASE_DIRS physica/field/include
        FILES physica/field/include/physica/field/device.cuh
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/field/field.ixx
)
target_include_directories(physica-field PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-field
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-math
)

# ============================================================================
# Fluid Grid
# ============================================================================
add_library(physica-fluids-grid STATIC)
target_sources(
        physica-fluids-grid
        PUBLIC
        FILE_SET headers TYPE HEADERS
        BASE_DIRS physica/fluids/grid/include
        FILES
        physica/fluids/grid/include/physica/fluids/grid/device.cuh
        physica/fluids/grid/include/physica/fluids/grid/interop.h
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
        physica-field
)

# ============================================================================
# Reconstruction Datasets
# ============================================================================
add_library(physica-dataset STATIC)
target_sources(
        physica-dataset
        PRIVATE
        physica/reconstruction/dataset/nerf_synthetic.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/reconstruction/dataset/multiview.ixx
        physica/reconstruction/dataset/nerf_synthetic.ixx
)
target_link_libraries(
        physica-dataset
        PUBLIC
        physica-math
        PRIVATE
        nlohmann_json::nlohmann_json
        physica::stb
)

# ============================================================================
# Instant-NGP
# ============================================================================
add_library(physica-instant-ngp STATIC)
target_sources(
        physica-instant-ngp
        PRIVATE
        physica/reconstruction/instant-ngp/instant_ngp.cpp
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
        physica/reconstruction/instant-ngp/instant_ngp.ixx
        physica/reconstruction/instant-ngp/network/network.ixx
        physica/reconstruction/instant-ngp/rendering/rendering.ixx
        physica/reconstruction/instant-ngp/sampling/sampling.ixx
        physica/reconstruction/instant-ngp/scene/scene.ixx
)
target_include_directories(physica-instant-ngp PUBLIC "${PHYSICA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        physica-instant-ngp
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        physica-dataset
        physica-math
        PRIVATE
        CUDA::cublas
        CUDA::cublasLt
        nlohmann_json::nlohmann_json
)

# ============================================================================
# L-BFGS-B
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
# Gas Fluids
# ============================================================================
add_library(physica-fluids-gas STATIC)
target_sources(
        physica-fluids-gas
        PRIVATE
        physica/fluids/gas/domain/domain.cpp
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
        BASE_DIRS physica/fluids/gas/include
        FILES
        physica/fluids/gas/include/physica/fluids/gas/device.cuh
        physica/fluids/gas/include/physica/fluids/gas/interop.h
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        physica/fluids/gas/domain/domain.ixx
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
# Liquid Fluids
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
        BASE_DIRS physica/fluids/liquid/include
        FILES
        physica/fluids/liquid/include/physica/fluids/liquid/device.cuh
        physica/fluids/liquid/include/physica/fluids/liquid/interop.h
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
        physica-field
        physica-fluids-grid
)

# ============================================================================
# Deformable Cloth
# ============================================================================
add_library(physica-deformables-cloth STATIC)
target_sources(
        physica-deformables-cloth
        PRIVATE
        physica/deformables/cloth/model/model.cpp
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
        physica/deformables/cloth/model/model.ixx
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
        physica-field
)

# ============================================================================
# Examples
# ============================================================================
if (PHYSICA_EXAMPLES)
    add_executable(
            physica-example-instant-ngp-cli
            examples/reconstruction/instant-ngp/main.cpp
    )
    target_link_libraries(
            physica-example-instant-ngp-cli
            PRIVATE
            physica-dataset
            physica-instant-ngp
    )

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
                physica-dataset
                physica-instant-ngp
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
