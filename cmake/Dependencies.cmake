include_guard(GLOBAL)

include(FetchContent)

FetchContent_Declare(
        nlohmann_json
        URL "https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz"
        URL_HASH SHA256=4B92EB0C06D10683F7447CE9406CB97CD4B453BE18D7279320F7B2F025C10187
        SYSTEM
        EXCLUDE_FROM_ALL
)

FetchContent_Declare(
        stb
        URL "https://github.com/nothings/stb/archive/28d546d5eb77d4585506a20480f4de2e706dff4c.tar.gz"
        URL_HASH SHA256=4EF16A0E174BC33887FEC582B01CA239155466E0B48081CC27304298556BED47
        SYSTEM
        EXCLUDE_FROM_ALL
)

FetchContent_Declare(
        cudnn_frontend
        URL "https://codeload.github.com/NVIDIA/cudnn-frontend/tar.gz/91159779637b672a3f768738f5b30eee33f2180d"
        URL_HASH SHA256=7ff261059996a163830007d8502664243b539e57ec72593dcb7e659d4604ce94
        SOURCE_SUBDIR physica-unused
        SYSTEM
        EXCLUDE_FROM_ALL
)

FetchContent_Declare(
        cutlass
        URL "https://codeload.github.com/NVIDIA/cutlass/tar.gz/dcf215af68a2d08d305076c152a06f201728cd53"
        URL_HASH SHA256=f77df767bdaccadee95989697913d414e7489e02c3d42aea609f68adc0627907
        SOURCE_SUBDIR physica-unused
        SYSTEM
        EXCLUDE_FROM_ALL
)

set(PHYSICA_DEPENDENCIES_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/dependencies")

macro(physica_require_dependency dependency)
    include("${PHYSICA_DEPENDENCIES_DIRECTORY}/${dependency}.cmake")
endmacro()
