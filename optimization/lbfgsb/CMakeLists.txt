# ============================================================================
# L-BFGS-B
# ============================================================================
add_library(physica-optimization-lbfgsb)
target_sources(
        physica-optimization-lbfgsb
        PRIVATE
        lbfgsb.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        lbfgsb.ixx
)
