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
option(BCP_BUILD_CONTROLLED_VALIDATION "Also build the diagnostic-only GIF_controlled validation executable (requires BCP_BUILD_GIF)" OFF)
if(BCP_BUILD_CONTROLLED_VALIDATION AND (NOT BCP_BUILD_GIF OR NOT UNIX))
    message(FATAL_ERROR "Controlled validation requires BCP_BUILD_GIF and a POSIX runtime")
endif()
if(APPLE)
    set(bcp_default_solver ACCELERATE)
else()
    set(bcp_default_solver MKL)
endif()
set(BCP_LINEAR_SOLVER "${bcp_default_solver}" CACHE STRING "Sparse solver: ACCELERATE, MKL, PANUA or EIGEN")
set_property(CACHE BCP_LINEAR_SOLVER PROPERTY STRINGS ACCELERATE MKL PANUA EIGEN)
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
        if(BCP_BUILD_CONTROLLED_VALIDATION)
            add_executable(bcp_controlled_io_test tests/controlled_io.cpp portable/ControlledValidation.cpp)
            target_include_directories(bcp_controlled_io_test PRIVATE portable)
            target_compile_definitions(bcp_controlled_io_test PRIVATE BCP_CONTROLLED_VALIDATION)
            add_test(NAME controlled_native_io COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/controlled_io.py"
                $<TARGET_FILE:bcp_controlled_io_test> "${CMAKE_BINARY_DIR}/controlled-tests")
            set_tests_properties(controlled_native_io PROPERTIES TIMEOUT 60)
            if(BCP_TEST_PIPELINE AND BCP_SCRIPT_ENGINE STREQUAL "OCTAVE" AND TARGET bcp_torch_energy_native)
                add_test(NAME controlled_runner_smoke COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/controlled_smoke.py"
                    $<TARGET_FILE:GIF_controlled> "${CMAKE_BINARY_DIR}/controlled-tests")
                set_tests_properties(controlled_runner_smoke PROPERTIES
                    ENVIRONMENT "BCP_BIN_DIR=$<TARGET_FILE_DIR:GIF_controlled>;BCP_OCTAVE_EXECUTABLE=${BCP_OCTAVE_EXECUTABLE};BCP_TORCH_NATIVE_DIR=$<TARGET_FILE_DIR:bcp_torch_energy_native>"
                    TIMEOUT 480)
            endif()
        endif()
        add_test(NAME python_unit COMMAND "${Python3_EXECUTABLE}" -m unittest discover
            -s "${PROJECT_SOURCE_DIR}/tests" -p "test_*.py" -v)
        if(BCP_SCRIPT_ENGINE STREQUAL "OCTAVE")
            if(BCP_OCTAVE_EXECUTABLE)
                set_property(TEST python_unit APPEND PROPERTY ENVIRONMENT
                    "BCP_OCTAVE_EXECUTABLE=${BCP_OCTAVE_EXECUTABLE}"
                    "BCP_TORCH_MODULE_DIR=${PROJECT_SOURCE_DIR}/BCP/GIF_and_CM")
            endif()
            if(TARGET bcp_torch_energy_native)
                set_property(TEST python_unit APPEND PROPERTY ENVIRONMENT
                    "BCP_TORCH_NATIVE_DIR=$<TARGET_FILE_DIR:bcp_torch_energy_native>")
            endif()
        endif()
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
            set(bcp_torch_test_env
                "BCP_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
                "BCP_OCTAVE_EXECUTABLE=${BCP_OCTAVE_EXECUTABLE}"
                "BCP_TORCH_MODULE_DIR=${PROJECT_SOURCE_DIR}/BCP/GIF_and_CM")
            if(TARGET bcp_torch_energy_native)
                list(APPEND bcp_torch_test_env
                    "BCP_TORCH_NATIVE_DIR=$<TARGET_FILE_DIR:bcp_torch_energy_native>")
            endif()
            add_test(NAME torch_energy_reference COMMAND "${BCP_OCTAVE_EXECUTABLE}"
                --no-gui --quiet "${PROJECT_SOURCE_DIR}/tests/torch_energy_reference.m")
            set_tests_properties(torch_energy_reference PROPERTIES
                ENVIRONMENT "${bcp_torch_test_env}" TIMEOUT 180)
            add_test(NAME matlab_sampling_reference COMMAND "${BCP_OCTAVE_EXECUTABLE}"
                --no-gui --quiet "${PROJECT_SOURCE_DIR}/tests/matlab_sampling_reference.m")
            set_tests_properties(matlab_sampling_reference PROPERTIES TIMEOUT 60)
            add_test(NAME controller_reference COMMAND "${BCP_OCTAVE_EXECUTABLE}"
                --no-gui --quiet "${PROJECT_SOURCE_DIR}/tests/controller_reference.m")
            set_tests_properties(controller_reference PROPERTIES
                ENVIRONMENT "${bcp_torch_test_env}" TIMEOUT 180)
            add_test(NAME full_pipeline_torch_cpu COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/full_pipeline.py")
            set_tests_properties(full_pipeline_torch_cpu PROPERTIES
                ENVIRONMENT "${bcp_torch_test_env};BCP_BIN_DIR=$<TARGET_FILE_DIR:GIF>;BCP_DEVICE=cpu;BCP_GIF_DEVICE=cpu;BCP_GIF_BACKEND=torch" TIMEOUT 1250)
            add_test(NAME compare_reference_diagnostics COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/scripts/compare.py" --mode reference --grid-size 10 --target-faces 60
                --trace --kkt-diagnostics)
            set_tests_properties(compare_reference_diagnostics PROPERTIES
                ENVIRONMENT "${bcp_torch_test_env};BCP_BIN_DIR=$<TARGET_FILE_DIR:GIF>;BCP_LINEAR_SOLVER=${BCP_LINEAR_SOLVER}"
                TIMEOUT 600)
            if(TARGET bcp_torch_energy_native)
                set_property(TEST full_pipeline APPEND PROPERTY ENVIRONMENT
                    "BCP_TORCH_NATIVE_DIR=$<TARGET_FILE_DIR:bcp_torch_energy_native>"
                    "BCP_TORCH_MODULE_DIR=${PROJECT_SOURCE_DIR}/BCP/GIF_and_CM")
            endif()
        endif()
    endif()
    add_executable(bcp_mkl_contract_test tests/mkl_contract.cpp portable/pardiso_mkl.cpp)
    target_include_directories(bcp_mkl_contract_test PRIVATE portable tests/mkl_stub)
    add_test(NAME mkl_adapter_contract COMMAND bcp_mkl_contract_test)
    if(BCP_LINEAR_SOLVER STREQUAL "ACCELERATE")
        add_executable(bcp_accelerate_test tests/accelerate_solver.cpp)
        target_link_libraries(bcp_accelerate_test PRIVATE bcp_linear_solver)
        add_test(NAME accelerate_solver COMMAND bcp_accelerate_test)
    endif()
    add_executable(bcp_cm_pattern_test tests/cm_pattern.cpp CompMajor/src/PardisoSolver.cpp)
    target_include_directories(bcp_cm_pattern_test PRIVATE CompMajor/src)
    target_link_libraries(bcp_cm_pattern_test PRIVATE bcp_linear_solver bcp_igl Threads::Threads)
    add_test(NAME cm_pattern COMMAND bcp_cm_pattern_test)
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
    set_property(TEST ${bcp_tests} APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1" "VECLIB_MAXIMUM_THREADS=1")
endif()
