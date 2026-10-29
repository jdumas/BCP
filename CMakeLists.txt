cmake_minimum_required(VERSION 3.24)
project(BCP LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Some pinned dependencies predate CMake 4.
set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
option(BCP_BUILD_GIF "Build the GIF stage" OFF)
set(BCP_LINEAR_SOLVER "MKL" CACHE STRING "Sparse solver: MKL, PANUA or EIGEN")
set_property(CACHE BCP_LINEAR_SOLVER PROPERTY STRINGS MKL PANUA EIGEN)
set(BCP_SCRIPT_ENGINE "MATLAB" CACHE STRING "GIF numerical runtime: MATLAB or OCTAVE")
set_property(CACHE BCP_SCRIPT_ENGINE PROPERTY STRINGS MATLAB OCTAVE)
option(BCP_BUILD_TESTS "Build portable native smoke tests" ON)
option(BCP_TEST_PIPELINE "Require live runtime and full Python integration tests" OFF)
if(BCP_TEST_PIPELINE AND (NOT BCP_BUILD_GIF OR NOT BCP_BUILD_TESTS))
    message(FATAL_ERROR "BCP_TEST_PIPELINE requires BCP_BUILD_GIF and BCP_BUILD_TESTS")
endif()
if(BCP_BUILD_TESTS)
    enable_testing()
endif()
include(cmake/dependencies.cmake)
find_package(Threads REQUIRED)
find_package(OpenMP)

add_library(bcp_mat INTERFACE)
target_include_directories(bcp_mat INTERFACE "${PROJECT_SOURCE_DIR}/portable")
target_link_libraries(bcp_mat INTERFACE Eigen3::Eigen matio::matio)

include(cmake/linear_solver.cmake)

add_executable(Parameterization_cmd
    CompMajor/src/main_cmd.cpp
    CompMajor/src/Energy.cpp
    CompMajor/src/EnergySymDir.cpp
    CompMajor/src/Newton.cpp
    CompMajor/src/PardisoSolver.cpp
    CompMajor/src/Solver.cpp
    CompMajor/src/Utils.cpp)
target_include_directories(Parameterization_cmd PRIVATE CompMajor/src)
target_compile_definitions(Parameterization_cmd PRIVATE BCP_PORTABLE _USE_MATH_DEFINES)
target_link_libraries(Parameterization_cmd PRIVATE bcp_igl bcp_mat bcp_linear_solver Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Parameterization_cmd PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(PT portable/pt_main.cpp)
target_link_libraries(PT PRIVATE bcp_mat)

if(BCP_BUILD_GIF)
    include(cmake/gif.cmake)
endif()

if(BCP_BUILD_TESTS)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME python_unit COMMAND "${Python3_EXECUTABLE}" -m unittest discover
            -s "${PROJECT_SOURCE_DIR}/tests" -p "test_*.py" -v)
    elseif(BCP_TEST_PIPELINE)
        message(FATAL_ERROR "The full pipeline test requires Python")
    endif()
    if(BCP_TEST_PIPELINE)
        if(NOT BCP_RUNTIME_TEST_AVAILABLE)
            message(FATAL_ERROR "The full pipeline test requires a real script runtime")
        endif()
        add_test(NAME full_pipeline COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/full_pipeline.py")
        set_tests_properties(full_pipeline PROPERTIES
            ENVIRONMENT "BCP_BIN_DIR=$<TARGET_FILE_DIR:GIF>;BCP_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}" TIMEOUT 1250)
        if(BCP_SCRIPT_ENGINE STREQUAL "OCTAVE")
            add_test(NAME torch_energy_reference COMMAND "${BCP_OCTAVE_EXECUTABLE}"
                --no-gui --quiet "${PROJECT_SOURCE_DIR}/tests/torch_energy_reference.m")
            set_tests_properties(torch_energy_reference PROPERTIES
                ENVIRONMENT "BCP_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}" TIMEOUT 180)
            add_test(NAME full_pipeline_torch_cpu COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/full_pipeline.py")
            set_tests_properties(full_pipeline_torch_cpu PROPERTIES
                ENVIRONMENT "BCP_BIN_DIR=$<TARGET_FILE_DIR:GIF>;BCP_PYTHON_EXECUTABLE=${Python3_EXECUTABLE};BCP_DEVICE=cpu;BCP_GIF_DEVICE=cpu;BCP_GIF_BACKEND=torch" TIMEOUT 1250)
        endif()
    endif()
    add_executable(bcp_mkl_contract_test tests/mkl_contract.cpp portable/pardiso_mkl.cpp)
    target_include_directories(bcp_mkl_contract_test PRIVATE portable tests/mkl_stub)
    add_test(NAME mkl_adapter_contract COMMAND bcp_mkl_contract_test)
    add_executable(bcp_native_test tests/native_smoke.cpp)
    target_link_libraries(bcp_native_test PRIVATE bcp_mat bcp_linear_solver bcp_igl)
    add_test(NAME native_smoke COMMAND bcp_native_test
        $<TARGET_FILE:Parameterization_cmd> $<TARGET_FILE:PT> "${CMAKE_BINARY_DIR}/smoke")
    set_tests_properties(native_smoke PROPERTIES TIMEOUT 60)
    add_test(NAME cm_usage COMMAND Parameterization_cmd)
    set_tests_properties(cm_usage PROPERTIES WILL_FAIL TRUE)
    add_test(NAME pt_usage COMMAND PT)
    set_tests_properties(pt_usage PROPERTIES WILL_FAIL TRUE)
    # Tiny correctness tests use one thread; production inherits user/runtime policy.
    get_property(bcp_tests DIRECTORY PROPERTY TESTS)
    set_property(TEST ${bcp_tests} APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
endif()
