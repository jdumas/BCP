# Build helper for the in-process Octave <-> CPython/Torch energy bridge.
#
# bcp_add_octave_torch_bridge(<target>) creates a MODULE library producing
# <target>.oct from portable/OctaveTorchBridge.cpp, embedding CPython + NumPy
# and linking the Octave libraries.  The bridge is POSIX-only (matching the
# optional Octave script backend) and is intentionally not wired into the GIF
# build here; the integration step includes this file and installs/wires the
# resulting module search path.
#
# Honors BCP_OCTAVE_CONFIG_EXECUTABLE / BCP_OCTAVE_EXECUTABLE when set so the
# same Octave as the script engine is targeted; otherwise octave-config is
# looked up on PATH (respecting CMAKE_PROGRAM_PATH).

include_guard(GLOBAL)

function(bcp_add_octave_torch_bridge target)
    if(NOT UNIX)
        message(FATAL_ERROR "The Octave/Torch bridge is POSIX-only, matching the Octave backend")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Embed NumPy)

    find_program(BCP_OCTAVE_CONFIG_EXECUTABLE NAMES octave-config
        HINTS ${BCP_OCTAVE_CONFIG_EXECUTABLE})
    if(NOT BCP_OCTAVE_CONFIG_EXECUTABLE AND BCP_OCTAVE_EXECUTABLE)
        get_filename_component(bcp_octave_bin "${BCP_OCTAVE_EXECUTABLE}" DIRECTORY)
        find_program(BCP_OCTAVE_CONFIG_EXECUTABLE NAMES octave-config
            HINTS "${bcp_octave_bin}" NO_DEFAULT_PATH)
    endif()
    if(NOT BCP_OCTAVE_CONFIG_EXECUTABLE)
        message(FATAL_ERROR "octave-config not found; set BCP_OCTAVE_CONFIG_EXECUTABLE")
    endif()

    foreach(var OCTINCLUDEDIR LIBDIR OCTLIBDIR)
        execute_process(
            COMMAND "${BCP_OCTAVE_CONFIG_EXECUTABLE}" -p ${var}
            OUTPUT_VARIABLE bcp_octave_${var}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY)
    endforeach()

    find_library(bcp_octinterp_lib NAMES octinterp
        HINTS "${bcp_octave_LIBDIR}" "${bcp_octave_OCTLIBDIR}" REQUIRED)
    find_library(bcp_octave_lib NAMES octave
        HINTS "${bcp_octave_LIBDIR}" "${bcp_octave_OCTLIBDIR}" REQUIRED)

    get_filename_component(bcp_bridge_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}" DIRECTORY)
    add_library(${target} MODULE "${bcp_bridge_dir}/portable/OctaveTorchBridge.cpp")
    set_target_properties(${target} PROPERTIES PREFIX "" SUFFIX ".oct")
    target_include_directories(${target} PRIVATE
        "${bcp_octave_OCTINCLUDEDIR}" ${Python3_NumPy_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE
        Python3::Python "${bcp_octinterp_lib}" "${bcp_octave_lib}")
    # The .oct is dlopen'ed by Octave from the build tree; let it find
    # libpython and the Octave libraries without environment variables.
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${bcp_octave_LIBDIR};${bcp_octave_OCTLIBDIR};${Python3_RUNTIME_LIBRARY_DIRS}")
endfunction()
