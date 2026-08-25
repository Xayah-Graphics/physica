# ============================================================================
# Smoke Example
# ============================================================================
add_library(physica-example-smoke)
target_sources(
        physica-example-smoke
        PRIVATE
        simulation.cu
        simulation_kernels.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        simulation.ixx
)
target_include_directories(physica-example-smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
        physica-example-smoke
        PUBLIC
        physica-fluids-gas-domain
        physica-fluids-gas-operators
        physica-fluids-gas-smoke
)

# ============================================================================
# Command-Line Example
# ============================================================================
add_executable(physica-example-smoke-cli main.cpp)
target_link_libraries(physica-example-smoke-cli PRIVATE physica-example-smoke)

# ============================================================================
# Spectra Visualization
# ============================================================================
if (PHYSICA_BUILD_SPECTRA)
    physica_require_dependency(spectra_sdk)
    spectra_add_provider(
            physica-example-smoke-spectra
            MODULE physica.example.fluids.gas.smoke.spectra
            TYPE physica::examples::smoke::Provider
    )
    target_sources(
            physica-example-smoke-spectra
            PRIVATE
            FILE_SET provider_module TYPE CXX_MODULES FILES
            spectra/provider.ixx
    )
    target_link_libraries(physica-example-smoke-spectra PRIVATE physica-example-smoke)
    set(bundle_directory "${PROJECT_BINARY_DIR}/spectra/smoke")
    set_target_properties(
            physica-example-smoke-spectra
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${bundle_directory}>"
    )
    add_custom_target(
            physica-example-smoke-spectra-bundle
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${bundle_directory}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/spectra/scene.spectra" "${bundle_directory}/scene.spectra"
            SOURCES spectra/scene.spectra
            VERBATIM
    )
    add_dependencies(physica-example-smoke-spectra physica-example-smoke-spectra-bundle)
endif ()
