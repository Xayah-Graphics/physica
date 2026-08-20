cmake_minimum_required(VERSION 4.4)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" PHYSICA_VERSION LIMIT_COUNT 1)
project(physica VERSION "${PHYSICA_VERSION}" LANGUAGES CXX CUDA)


# ============================================================================
# Project Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PHYSICA_BUILD_INSTANT_NGP "Build instant-ngp" ON)

include(cmake/Dependencies.cmake)
physica_require_dependency(cccl)

if (MSVC)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()

add_subdirectory(reconstruction)

if (PHYSICA_BUILD_INSTANT_NGP)
    add_executable(physica main.cpp)
    target_link_libraries(physica PRIVATE physica-instant-ngp physica-dataset-nerf-synthetic)
    set_property(TARGET physica PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(physica PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
endif ()
