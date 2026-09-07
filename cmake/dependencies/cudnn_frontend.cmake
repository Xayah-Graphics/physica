include_guard(GLOBAL)

FetchContent_MakeAvailable(cudnn_frontend)
add_library(physica-cudnn-frontend INTERFACE)
add_library(physica::cudnn-frontend ALIAS physica-cudnn-frontend)
target_include_directories(physica-cudnn-frontend SYSTEM INTERFACE "${cudnn_frontend_SOURCE_DIR}/include")
target_link_libraries(physica-cudnn-frontend INTERFACE cuDNN::cuDNN nlohmann_json::nlohmann_json)
