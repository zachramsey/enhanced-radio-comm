
cmake_minimum_required(VERSION 3.15)
project(remote_runner VERSION 1.0.0)

# Define the library
add_library(remote_runner SHARED src/remote_runner.cpp)
add_library(RemoteRunner::remote_runner ALIAS remote_runner)

# Public include directory for this library
target_include_directories(remote_runner PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link against Executorch
target_link_libraries(remote_runner PUBLIC 
    executorch
    extension_module_static
    extension_tensor
    optimized_native_cpu_ops_lib
    quantized_ops_lib
    xnnpack_backend
)

# --- Installation ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the library
install(TARGETS remote_runner
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT remote_runner_lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT remote_runner_lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT remote_runner_lib
)

# Install public header files
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT remote_runner_dev
    FILES_MATCHING PATTERN "*.h"
)

# Generate and install the package configuration file (RemoteRunnerConfig.cmake)
# This allows `find_package(remote_runner)` to work.

# Configure the package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RemoteRunnerConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/RemoteRunnerConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/remote_runner/cmake"
    PATH_VARS CMAKE_INSTALL_PREFIX CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
)

# Generate the version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/RemoteRunnerConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Install the CMake configuration files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/RemoteRunnerConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/RemoteRunnerConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/remote_runner/cmake"
    COMPONENT remote_runner_dev
)

#===================================================================================================

# # remote_runner/CMakeLists.txt
# cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# project(remote_runner CXX)

# # --- Define Library Sources ---
# set(REMOTE_RUNNER_SOURCES src/remote_runner.cpp)
# set(REMOTE_RUNNER_PUBLIC_HEADERS include/remote_runner.h)

# # --- Define Library Target ---
# # Inherits BUILD_SHARED_LIBS from parent scope
# add_library(remote_runner SHARED ${REMOTE_RUNNER_SOURCES} ${REMOTE_RUNNER_PUBLIC_HEADERS})
# add_library(MonkeyComms::RemoteRunner ALIAS remote_runner)

# # --- Configure Include Directories ---
# # PUBLIC includes point to this library's include dir
# # PRIVATE includes for src dir
# # ET includes come from linked targets automatically
# target_include_directories(
#     remote_runner
#     PUBLIC
#         $<INSTALL_INTERFACE:include>
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/src
# )

# # --- Link Dependencies ---
# # Link against targets built by the parent scope's add_subdirectory(executorch)
# target_link_libraries(
#     remote_runner
#     PRIVATE 
#         executorch
#         extension_module_static
#         extension_tensor
#         optimized_native_cpu_ops_lib
#         quantized_ops_lib
#         xnnpack_backend
# )

# # Install the library
# install(
#     TARGETS remote_runner
#     EXPORT remote_runner-targets
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#     INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )

# # Install the headers
# install(
#     FILES ${REMOTE_RUNNER_PUBLIC_HEADERS}
#     DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )

# # Generate and install CMake package config files

# # Version file
# write_basic_package_version_file(
#     "remote_runner-config_version.cmake"
#     VERSION 1.0.0
#     COMPATIBILITY SameMajorVersion
# )

# # Config file
# configure_package_config_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/remote_runner-config.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/remote_runner-config.cmake"
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/remote_runner
# )

# # Install the config files
# install(
#     FILES
#         "${CMAKE_CURRENT_BINARY_DIR}/remote_runner-config.cmake"
#         "${CMAKE_CURRENT_BINARY_DIR}/remote_runner-config_version.cmake"
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/remote_runner
# )

# # Install the export file
# install(
#     EXPORT remote_runner-targets
#     FILE remote_runner-targets.cmake
#     NAMESPACE MonkeyComms::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/remote_runner
# )