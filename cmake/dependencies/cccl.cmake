include_guard(GLOBAL)

physica_require_dependency(cuda)

find_package(
        CCCL 3.3.4 EXACT REQUIRED CONFIG GLOBAL
        COMPONENTS libcudacxx
        PATHS "${CUDAToolkit_LIBRARY_ROOT}/lib/cmake/cccl"
        NO_DEFAULT_PATH
)
