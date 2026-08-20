physica_require_dependency(nlohmann_json)
physica_require_dependency(stb)

add_library(physica-dataset-multiview)
target_sources(
        physica-dataset-multiview
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        multiview.ixx
)
set_property(TARGET physica-dataset-multiview PROPERTY CXX_MODULE_STD ON)

add_library(physica-dataset-nerf-synthetic)
target_sources(
        physica-dataset-nerf-synthetic
        PRIVATE
        nerf_synthetic.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        nerf_synthetic.ixx
)
target_link_libraries(
        physica-dataset-nerf-synthetic
        PUBLIC
        physica-dataset-multiview
        PRIVATE
        nlohmann_json::nlohmann_json
        physica::stb
)
set_property(TARGET physica-dataset-nerf-synthetic PROPERTY CXX_MODULE_STD ON)
