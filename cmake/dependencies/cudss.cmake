include_guard(GLOBAL)

physica_require_dependency(cuda)

find_package(
        cudss 0.8.0 EXACT REQUIRED CONFIG GLOBAL
        COMPONENTS cudss
        PATHS "C:/Program Files/NVIDIA cuDSS/v0.8/lib/13/cmake/cudss"
)
