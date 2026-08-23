# ============================================================================
# Keyframe Smoke Letter Experiment
# ============================================================================
physica_require_dependency(stb)

add_library(physica-example-keyframe-smoke)
target_sources(
        physica-example-keyframe-smoke
        PRIVATE
        experiment.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        experiment.ixx
)
target_link_libraries(
        physica-example-keyframe-smoke
        PUBLIC
        physica-fluids-gas-keyframe-smoke
        PRIVATE
        physica::stb
)

add_executable(physica-example-keyframe-smoke-cli main.cpp)
target_link_libraries(physica-example-keyframe-smoke-cli PRIVATE physica-example-keyframe-smoke)
