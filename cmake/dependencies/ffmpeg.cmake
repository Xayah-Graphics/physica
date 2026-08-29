include_guard(GLOBAL)

if (WIN32)
    FetchContent_Declare(
            ffmpeg
            URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-08-20-13-45/ffmpeg-n8.1.2-44-g7c533d0f86-win64-lgpl-shared-8.1.zip"
            URL_HASH SHA256=D311C8C7B86E06B54588E442652F963BAE165BD4D8393E73CC9EBB445B025547
            SYSTEM
            EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(ffmpeg)

    add_library(FFmpeg::avcodec SHARED IMPORTED GLOBAL)
    set_target_properties(
            FFmpeg::avcodec
            PROPERTIES
            IMPORTED_IMPLIB "${ffmpeg_SOURCE_DIR}/lib/avcodec.lib"
            IMPORTED_LOCATION "${ffmpeg_SOURCE_DIR}/bin/avcodec-62.dll"
            INTERFACE_INCLUDE_DIRECTORIES "${ffmpeg_SOURCE_DIR}/include"
    )

    add_library(FFmpeg::avformat SHARED IMPORTED GLOBAL)
    set_target_properties(
            FFmpeg::avformat
            PROPERTIES
            IMPORTED_IMPLIB "${ffmpeg_SOURCE_DIR}/lib/avformat.lib"
            IMPORTED_LOCATION "${ffmpeg_SOURCE_DIR}/bin/avformat-62.dll"
            INTERFACE_INCLUDE_DIRECTORIES "${ffmpeg_SOURCE_DIR}/include"
    )

    add_library(FFmpeg::avutil SHARED IMPORTED GLOBAL)
    set_target_properties(
            FFmpeg::avutil
            PROPERTIES
            IMPORTED_IMPLIB "${ffmpeg_SOURCE_DIR}/lib/avutil.lib"
            IMPORTED_LOCATION "${ffmpeg_SOURCE_DIR}/bin/avutil-60.dll"
            INTERFACE_INCLUDE_DIRECTORIES "${ffmpeg_SOURCE_DIR}/include"
    )

    add_library(FFmpeg::swscale SHARED IMPORTED GLOBAL)
    set_target_properties(
            FFmpeg::swscale
            PROPERTIES
            IMPORTED_IMPLIB "${ffmpeg_SOURCE_DIR}/lib/swscale.lib"
            IMPORTED_LOCATION "${ffmpeg_SOURCE_DIR}/bin/swscale-9.dll"
            INTERFACE_INCLUDE_DIRECTORIES "${ffmpeg_SOURCE_DIR}/include"
    )

    add_library(FFmpeg::FFmpeg INTERFACE IMPORTED GLOBAL)
    target_link_libraries(
            FFmpeg::FFmpeg
            INTERFACE
            FFmpeg::avformat
            FFmpeg::avcodec
            FFmpeg::swscale
            FFmpeg::avutil
    )
else ()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFmpeg REQUIRED IMPORTED_TARGET libavformat libavcodec libswscale libavutil)
    add_library(FFmpeg::FFmpeg ALIAS PkgConfig::FFmpeg)
endif ()
