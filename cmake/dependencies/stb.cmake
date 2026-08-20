include_guard(GLOBAL)

FetchContent_MakeAvailable(stb)

add_library(physica-stb INTERFACE)
add_library(physica::stb ALIAS physica-stb)
target_include_directories(physica-stb SYSTEM INTERFACE "${stb_SOURCE_DIR}")
