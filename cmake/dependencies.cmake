include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/CPM.cmake")

# Header-only dependencies: do not configure their examples, tests or submodules.
CPMAddPackage(NAME eigen VERSION 3.4.0
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    DOWNLOAD_ONLY YES)
add_library(bcp_eigen INTERFACE)
target_include_directories(bcp_eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")
add_library(Eigen3::Eigen ALIAS bcp_eigen)

# Preserve the libigl revision used by the original Visual Studio projects.
CPMAddPackage(NAME libigl
    URL https://github.com/Roipo/libigl/archive/5a4863f7c77e6fa8eed5830e01f455b01afb4538.tar.gz
    URL_HASH SHA256=9be39f1f538349fed6d773579d5845d6ea137ea53a606824884cc73c80202049
    DOWNLOAD_ONLY YES)
add_library(bcp_igl INTERFACE)
target_include_directories(bcp_igl SYSTEM INTERFACE "${libigl_SOURCE_DIR}/include")
target_link_libraries(bcp_igl INTERFACE Eigen3::Eigen)

CPMAddPackage(NAME zlib VERSION 1.3.1
    URL https://github.com/madler/zlib/archive/refs/tags/v1.3.1.tar.gz
    URL_HASH SHA256=17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c
    EXCLUDE_FROM_ALL YES
    OPTIONS "ZLIB_BUILD_EXAMPLES OFF")
add_library(ZLIB::ZLIB ALIAS zlibstatic)
add_library(zlib-static ALIAS zlibstatic)
# zlib's generated zconf.h is in the build tree.
target_include_directories(zlibstatic INTERFACE "${zlib_SOURCE_DIR}" "${zlib_BINARY_DIR}")

CPMAddPackage(NAME matio VERSION 1.5.28
    URL https://github.com/tbeu/matio/archive/refs/tags/v1.5.28.tar.gz
    URL_HASH SHA256=04d14160a637ea822593c336b231227372179f650250c98024a8a2b744afef25
    EXCLUDE_FROM_ALL YES
    OPTIONS "MATIO_SHARED OFF" "MATIO_MAT73 OFF" "MATIO_WITH_HDF5 OFF"
            "MATIO_WITH_ZLIB ON" "MATIO_BUILD_TESTING OFF" "HDF5_USE_STATIC_LIBRARIES ON")
