# ============================================================================
# Command-Line Example
# ============================================================================
add_executable(physica-example-instant-ngp-cli main.cpp)
target_link_libraries(
        physica-example-instant-ngp-cli
        PRIVATE
        physica-instant-ngp
        physica-dataset-nerf-synthetic
)

# ============================================================================
# Spectra Visualization
# ============================================================================
if (PHYSICA_BUILD_SPECTRA)
    physica_require_dependency(spectra_sdk)
    spectra_add_provider(
            physica-example-instant-ngp-spectra
            MODULE physica.example.reconstruction.instant_ngp.spectra
            TYPE physica::examples::instant_ngp::Provider
    )
    target_sources(
            physica-example-instant-ngp-spectra
            PRIVATE
            FILE_SET provider_module TYPE CXX_MODULES FILES
            spectra/provider.ixx
    )
    target_link_libraries(
            physica-example-instant-ngp-spectra
            PRIVATE
            physica-instant-ngp
            physica-dataset-nerf-synthetic
    )
    set(bundle_directory "${PROJECT_BINARY_DIR}/spectra/instant-ngp")
    set_target_properties(
            physica-example-instant-ngp-spectra
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${bundle_directory}>"
    )
    add_custom_target(
            physica-example-instant-ngp-spectra-bundle
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${bundle_directory}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/spectra/scene.spectra" "${bundle_directory}/scene.spectra"
            SOURCES spectra/scene.spectra
            VERBATIM
    )
    add_dependencies(physica-example-instant-ngp-spectra physica-example-instant-ngp-spectra-bundle)
endif ()
