# Keep the bootstrap and all downloaded sources inside the build tree.
set(CPM_DOWNLOAD_VERSION 0.42.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
set(CPM_BOOTSTRAP_SHA256 2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a)
set(cpm_installed_hash "")
if(EXISTS "${CPM_DOWNLOAD_LOCATION}")
    file(SHA256 "${CPM_DOWNLOAD_LOCATION}" cpm_installed_hash)
endif()
if(NOT cpm_installed_hash STREQUAL CPM_BOOTSTRAP_SHA256)
    file(DOWNLOAD
        "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
        "${CPM_DOWNLOAD_LOCATION}"
        EXPECTED_HASH SHA256=${CPM_BOOTSTRAP_SHA256}
        TLS_VERIFY ON
    )
endif()
include("${CPM_DOWNLOAD_LOCATION}")
