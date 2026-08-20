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

# ============================================================================
# Compiler Configuration
# ============================================================================
if (MSVC)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()

# ============================================================================
# Targets
# ============================================================================
add_subdirectory(reconstruction/dataset)
add_subdirectory(reconstruction/instant-ngp)
add_subdirectory(fluids/gas/smoke)
add_subdirectory(fluids/liquid/particle)
add_subdirectory(deformables/cloth)
if (PHYSICA_EXAMPLES)
    add_subdirectory(examples/reconstruction/instant-ngp)
    add_subdirectory(examples/fluids/gas/smoke)
    add_subdirectory(examples/fluids/liquid/pbf-dam-break)
    add_subdirectory(examples/deformables/cloth)
endif ()
