# control_runner/CMakeLists.txt (when used with Top-Level CMake)
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(control_runner CXX)

# --- Define Library Sources ---
set(CONTROL_RUNNER_SOURCES src/control_runner.cpp)
set(CONTROL_RUNNER_PUBLIC_HEADERS include/control_runner.h)

# --- Define Library Target ---
# Inherits BUILD_SHARED_LIBS from parent scope
add_library(control_runner SHARED ${CONTROL_RUNNER_SOURCES} ${CONTROL_RUNNER_PUBLIC_HEADERS})
add_library(MonkeyComms::ControlRunner ALIAS control_runner)

# --- Configure Include Directories ---
# PUBLIC includes point to this library's include dir
# PRIVATE includes for src dir
# ET includes come from linked targets automatically
target_include_directories(
    control_runner
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# --- Link Dependencies ---
# Link against targets built by the parent scope's add_subdirectory(executorch)
target_link_libraries(
    control_runner
    PRIVATE 
        executorch
        extension_module_static
        extension_tensor
        optimized_native_cpu_ops_lib
        quantized_ops_lib
        xnnpack_backend
)

# Install the library
install(
    TARGETS control_runner
    EXPORT control_runner-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the headers
install(
    FILES ${CONTROL_RUNNER_PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Generate and install CMake package config files

# Version file
write_basic_package_version_file(
    "control_runner-config_version.cmake"
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

# Config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/control_runner-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/control_runner-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/control_runner
)

# Install the config files
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/control_runner-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/control_runner-config_version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/control_runner
)

# Install the export file
install(
    EXPORT control_runner-targets
    FILE control_runner-targets.cmake
    NAMESPACE MonkeyComms::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/control_runner
)
