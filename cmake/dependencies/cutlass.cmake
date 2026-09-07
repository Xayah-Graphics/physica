include_guard(GLOBAL)

FetchContent_MakeAvailable(cutlass)
add_library(physica-cutlass INTERFACE)
add_library(physica::cutlass ALIAS physica-cutlass)
target_include_directories(physica-cutlass SYSTEM INTERFACE
        "${cutlass_SOURCE_DIR}/include"
        "${cutlass_SOURCE_DIR}/examples/45_dual_gemm")
