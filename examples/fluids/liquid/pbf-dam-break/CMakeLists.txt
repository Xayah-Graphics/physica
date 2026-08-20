# ============================================================================
# PBF Dam-Break Example
# ============================================================================
add_library(physica-example-pbf-dam-break)
target_sources(
        physica-example-pbf-dam-break
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        simulation.ixx
)
target_link_libraries(physica-example-pbf-dam-break PUBLIC physica-fluids-liquid-particle)

# ============================================================================
# Command-Line Example
# ============================================================================
add_executable(physica-example-pbf-dam-break-cli main.cpp)
target_link_libraries(physica-example-pbf-dam-break-cli PRIVATE physica-example-pbf-dam-break)

# ============================================================================
# Spectra Visualization
# ============================================================================
if (PHYSICA_BUILD_SPECTRA)
    physica_require_dependency(spectra_sdk)
    spectra_add_provider(
            physica-example-pbf-dam-break-spectra
            MODULE physica.example.fluids.liquid.pbf_dam_break.spectra
            TYPE physica::examples::pbf_dam_break::Provider
    )
    target_sources(
            physica-example-pbf-dam-break-spectra
            PRIVATE
            spectra/kernels.cu
            spectra/kernels.h
            PRIVATE
            FILE_SET provider_module TYPE CXX_MODULES FILES
            spectra/provider.ixx
    )
    target_link_libraries(physica-example-pbf-dam-break-spectra PRIVATE physica-example-pbf-dam-break)
    set(bundle_directory "${PROJECT_BINARY_DIR}/spectra/pbf-dam-break")
    set_target_properties(
            physica-example-pbf-dam-break-spectra
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${bundle_directory}>"
    )
    add_custom_target(
            physica-example-pbf-dam-break-spectra-bundle
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${bundle_directory}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/spectra/scene.spectra" "${bundle_directory}/scene.spectra"
            SOURCES spectra/scene.spectra
            VERBATIM
    )
    add_dependencies(physica-example-pbf-dam-break-spectra physica-example-pbf-dam-break-spectra-bundle)
endif ()
