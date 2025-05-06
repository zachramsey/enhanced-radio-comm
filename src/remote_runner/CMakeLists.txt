# remote_runner/CMakeLists.txt
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(remote_runner CXX)

# --- Define Library Sources ---
set(REMOTE_RUNNER_SOURCES src/remote_runner.cpp)
set(REMOTE_RUNNER_PUBLIC_HEADERS include/remote_runner.h)

# --- Define Library Target ---
# Inherits BUILD_SHARED_LIBS from parent scope
add_library(remote_runner SHARED ${REMOTE_RUNNER_SOURCES} ${REMOTE_RUNNER_PUBLIC_HEADERS})
add_library(MonkeyComms::RemoteRunner ALIAS remote_runner)

# --- Configure Include Directories ---
# PUBLIC includes point to this library's include dir
# PRIVATE includes for src dir
# ET includes come from linked targets automatically
target_include_directories(
    remote_runner
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# --- Link Dependencies ---
# Link against targets built by the parent scope's add_subdirectory(executorch)
target_link_libraries(
    remote_runner
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
    TARGETS remote_runner
    EXPORT remote_runner-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the headers
install(
    FILES ${REMOTE_RUNNER_PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Generate and install CMake package config files

# Version file
write_basic_package_version_file(
    "remote_runner-config_version.cmake"
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

# Config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/remote_runner-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/remote_runner-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/remote_runner
)

# Install the config files
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/remote_runner-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/remote_runner-config_version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/remote_runner
)

# Install the export file
install(
    EXPORT remote_runner-targets
    FILE remote_runner-targets.cmake
    NAMESPACE MonkeyComms::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/remote_runner
)