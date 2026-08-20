include_guard(GLOBAL)

find_package(SpectraSDK CONFIG REQUIRED GLOBAL)
set(CACHE{SpectraSDK_INTERNAL_DIRECTORY}
        TYPE INTERNAL
        HELP "Spectra SDK Provider support files"
        FORCE
        VALUE "${SpectraSDK_INTERNAL_DIRECTORY}"
)
