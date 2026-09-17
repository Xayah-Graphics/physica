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

set(PHYSICA_DEPENDENCIES_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/dependencies")

macro(physica_require_dependency dependency)
    include("${PHYSICA_DEPENDENCIES_DIRECTORY}/${dependency}.cmake")
endmacro()
