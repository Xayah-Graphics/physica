# ============================================================================
# Adjoint Fluid Control Bunny Experiment
# ============================================================================
physica_require_dependency(stb)

add_library(physica-example-adjoint-control)
target_sources(
        physica-example-adjoint-control
        PRIVATE
        experiment.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        experiment.ixx
)
target_link_libraries(
        physica-example-adjoint-control
        PUBLIC
        physica-fluids-gas-adjoint-control
        PRIVATE
        physica::stb
)

add_executable(physica-example-adjoint-control-cli main.cpp)
target_link_libraries(physica-example-adjoint-control-cli PRIVATE physica-example-adjoint-control)
