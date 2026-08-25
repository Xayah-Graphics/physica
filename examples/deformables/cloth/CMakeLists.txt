# ============================================================================
# Cloth Example
# ============================================================================
add_library(physica-example-cloth)
target_sources(
        physica-example-cloth
        PRIVATE
        simulation.cu
        simulation_kernels.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        simulation.ixx
)
target_include_directories(physica-example-cloth PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
        physica-example-cloth
        PUBLIC
        physica-deformables-cloth-explicit-dynamics
        physica-deformables-cloth-operators
)

# ============================================================================
# Command-Line Example
# ============================================================================
add_executable(physica-example-cloth-cli main.cpp)
target_link_libraries(physica-example-cloth-cli PRIVATE physica-example-cloth)

# ============================================================================
# Spectra Visualization
# ============================================================================
if (PHYSICA_BUILD_SPECTRA)
    physica_require_dependency(spectra_sdk)
    spectra_add_provider(
            physica-example-cloth-spectra
            MODULE physica.example.deformables.cloth.spectra
            TYPE physica::examples::cloth::Provider
    )
    target_sources(
            physica-example-cloth-spectra
            PRIVATE
            spectra/kernels.cu
            spectra/kernels.h
            PRIVATE
            FILE_SET provider_module TYPE CXX_MODULES FILES
            spectra/provider.ixx
    )
    target_link_libraries(physica-example-cloth-spectra PRIVATE physica-example-cloth)
    set(bundle_directory "${PROJECT_BINARY_DIR}/spectra/cloth")
    set_target_properties(
            physica-example-cloth-spectra
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${bundle_directory}>"
    )
    add_custom_target(
            physica-example-cloth-spectra-bundle
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${bundle_directory}/assets"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/spectra/scene.spectra" "${bundle_directory}/scene.spectra"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/spectra/assets/flag.ply" "${bundle_directory}/assets/flag.ply"
            SOURCES
            spectra/scene.spectra
            spectra/assets/flag.ply
            VERBATIM
    )
    add_dependencies(physica-example-cloth-spectra physica-example-cloth-spectra-bundle)
endif ()
