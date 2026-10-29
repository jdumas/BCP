# Reuse the original MATLAB numerical scripts. Keep the direct MATLAB engine
# separate from the optional Octave compatibility backend.
# CGAL 5.6 is not compatible with arbitrary newer Boost releases.
CPMAddPackage(NAME boost VERSION 1.83.0
    URL https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.bz2
    URL_HASH SHA256=6478edfe2f3305127cffe8caf73ea0176c53769f4bf1585be237eb30798c3b8e
    DOWNLOAD_ONLY YES)
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()
set(Boost_NO_BOOST_CMAKE ON)
set(Boost_NO_SYSTEM_PATHS ON)
set(BOOST_ROOT "${boost_SOURCE_DIR}")
set(Boost_INCLUDE_DIR "${boost_SOURCE_DIR}")
find_package(Boost 1.83 EXACT REQUIRED)
# 5.6.2+ fixes invalid iterator base() calls rejected by modern Clang.
CPMAddPackage(NAME cgal VERSION 5.6.3
    URL https://github.com/CGAL/cgal/releases/download/v5.6.3/CGAL-5.6.3.tar.xz
    URL_HASH SHA256=15c743cb395d1a0855b9062525f3ae0cd40486489acfe7ce1457c3710ab34111
    DOWNLOAD_ONLY YES)
find_package(CGAL 5.6 REQUIRED CONFIG PATHS "${cgal_SOURCE_DIR}" NO_DEFAULT_PATH)
CPMAddPackage(NAME gmm VERSION 4.2
    URL https://download-mirror.savannah.gnu.org/releases/getfem/stable/gmm-4.2.tar.gz
    URL_HASH SHA256=bc94080c9b2e1da2f8c886cfafa490bc1d5640912e6342ddd457c789f577fdb6
    DOWNLOAD_ONLY YES)
CPMAddPackage(NAME nlopt VERSION 2.10.0
    URL https://github.com/stevengj/nlopt/archive/refs/tags/v2.10.0.tar.gz
    URL_HASH SHA256=506f83a9e778ad4f204446e99509cb2bdf5539de8beccc260a014bd560237be1
    EXCLUDE_FROM_ALL YES
    OPTIONS "NLOPT_PYTHON OFF" "NLOPT_OCTAVE OFF" "NLOPT_MATLAB OFF" "NLOPT_GUILE OFF"
            "NLOPT_SWIG OFF" "NLOPT_TESTS OFF" "BUILD_SHARED_LIBS OFF")
set(gif_root "${PROJECT_SOURCE_DIR}/GIF/GIF")
file(GLOB gif_energy CONFIGURE_DEPENDS "${gif_root}/Isometric_Injective_Energies/src/*.cpp")
# Standalone demo I/O is not used by GIF (and was absent from its VS source list).
list(FILTER gif_energy EXCLUDE REGEX "/IO\\.cpp$")
file(GLOB gif_parser CONFIGURE_DEPENDS "${gif_root}/Parser/*.cpp")
string(TOUPPER "${BCP_SCRIPT_ENGINE}" BCP_SCRIPT_ENGINE)
set(BCP_RUNTIME_TEST_AVAILABLE TRUE)
if(BCP_SCRIPT_ENGINE STREQUAL "MATLAB")
    find_package(Matlab REQUIRED COMPONENTS ENG_LIBRARY)
    add_library(bcp_script_engine STATIC "${gif_root}/Utils/MatlabInterface.cpp")
    target_link_libraries(bcp_script_engine PUBLIC Matlab::eng Matlab::mx)
    target_compile_definitions(bcp_script_engine PRIVATE MATLAB_DEFAULT_RELEASE=R2018a)
elseif(BCP_SCRIPT_ENGINE STREQUAL "OCTAVE")
    if(NOT UNIX)
        message(FATAL_ERROR "The optional Octave bridge is POSIX-only; use MATLAB on Windows")
    endif()
    add_library(bcp_script_engine STATIC portable/OctaveInterface.cpp)
    find_program(BCP_OCTAVE_EXECUTABLE NAMES octave-cli octave)
    if(NOT BCP_OCTAVE_EXECUTABLE)
        set(BCP_RUNTIME_TEST_AVAILABLE FALSE)
        message(WARNING "Octave runtime absent: its numerical test will be explicitly SKIPPED, not counted as passed")
    endif()
else()
    message(FATAL_ERROR "Unknown BCP_SCRIPT_ENGINE=${BCP_SCRIPT_ENGINE}; choose MATLAB or OCTAVE")
endif()
target_include_directories(bcp_script_engine PUBLIC "${gif_root}")
target_link_libraries(bcp_script_engine PUBLIC bcp_mat)
add_executable(GIF
    portable/gif_main.cpp
    "${gif_root}/GIF/GIF.cpp"
    "${gif_root}/CGAL/CGAL_Borders.cpp"
    "${gif_root}/CGAL/CGAL_UTIL.cpp"
    "${gif_root}/Utils/CompilationAccelerator.cpp"
    "${gif_root}/Utils/Limit_Util.cpp"
    "${gif_root}/Utils/logger.cpp"
    "${gif_root}/Utils/logging.cpp"
    "${gif_root}/Utils/MatlabGMMDataExchange.cpp"
    "${gif_root}/Utils/MeshAlgorithms.cpp"
    "${gif_root}/Utils/PardisoLinearSolver.cpp"
    "${gif_root}/Utils/triangle.c"
    "${gif_root}/Utils/triangulation.cpp"
    "${gif_root}/Utils/Utilities.cpp"
    ${gif_energy} ${gif_parser})
target_include_directories(GIF PRIVATE "${gif_root}" "${gif_root}/Utils" "${gmm_SOURCE_DIR}/include")
target_include_directories(GIF SYSTEM BEFORE PRIVATE "${boost_SOURCE_DIR}")
target_compile_definitions(GIF PRIVATE TRILIBRARY ANSI_DECLARATORS _USE_MATH_DEFINES NOMINMAX
    BCP_MATLAB_SCRIPTS_DIR="${gif_root}/MatlabScripts")
# GMM 4.2 still uses the removed C++ register storage-class specifier.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(GIF PRIVATE -Wno-register)
endif()
# GMM 4.2 predates C++17 and still uses std::binary_function on MSVC.
if(MSVC)
    target_compile_definitions(GIF PRIVATE _HAS_AUTO_PTR_ETC=1)
endif()
target_link_libraries(GIF PRIVATE bcp_igl bcp_mat bcp_linear_solver bcp_script_engine CGAL::CGAL Boost::headers nlopt Threads::Threads)
# Legacy code has large template translation units; don't run many of them at once.
set_property(GLOBAL APPEND PROPERTY JOB_POOLS bcp_gif_compile=2)
set_property(TARGET GIF PROPERTY JOB_POOL_COMPILE bcp_gif_compile)
if(BCP_SCRIPT_ENGINE STREQUAL "OCTAVE" AND BCP_OCTAVE_EXECUTABLE)
    target_compile_definitions(GIF PRIVATE BCP_DEFAULT_OCTAVE_EXECUTABLE="${BCP_OCTAVE_EXECUTABLE}")
endif()
if(BCP_BUILD_TESTS)
    if(BCP_SCRIPT_ENGINE STREQUAL "OCTAVE")
        add_executable(bcp_octave_missing_test tests/octave_missing.cpp)
        target_link_libraries(bcp_octave_missing_test PRIVATE bcp_script_engine)
        add_test(NAME octave_missing_runtime COMMAND bcp_octave_missing_test "${CMAKE_BINARY_DIR}/missing-octave-test")
        set_tests_properties(octave_missing_runtime PROPERTIES TIMEOUT 10)
    endif()
    add_executable(bcp_script_engine_test tests/script_engine_smoke.cpp)
    target_link_libraries(bcp_script_engine_test PRIVATE bcp_script_engine)
    target_compile_definitions(bcp_script_engine_test PRIVATE BCP_MATLAB_SCRIPTS_DIR="${gif_root}/MatlabScripts")
    if(NOT BCP_RUNTIME_TEST_AVAILABLE)
        target_compile_definitions(bcp_script_engine_test PRIVATE BCP_SKIP_SCRIPT_TEST)
    endif()
    if(BCP_SCRIPT_ENGINE STREQUAL "OCTAVE")
        target_compile_definitions(bcp_script_engine_test PRIVATE BCP_OCTAVE_ENGINE)
    endif()
    add_test(NAME script_engine_numerics COMMAND bcp_script_engine_test)
    set_tests_properties(script_engine_numerics PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    if(BCP_SCRIPT_ENGINE STREQUAL "OCTAVE" AND BCP_OCTAVE_EXECUTABLE)
        set_tests_properties(script_engine_numerics PROPERTIES ENVIRONMENT "BCP_OCTAVE_EXECUTABLE=${BCP_OCTAVE_EXECUTABLE}")
    endif()
    add_test(NAME gif_usage COMMAND GIF)
    set_tests_properties(gif_usage PROPERTIES WILL_FAIL TRUE)
endif()
