
cmake_minimum_required(VERSION 3.15)
project(control_runner VERSION 1.0.0)

# Define the library
add_library(control_runner SHARED src/control_runner.cpp)
add_library(ControlRunner::control_runner ALIAS control_runner)

# Public include directory for this library
target_include_directories(control_runner PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link against Executorch
target_link_libraries(control_runner PUBLIC 
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
install(TARGETS control_runner
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT control_runner_lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT control_runner_lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT control_runner_lib
)

# Install public header files
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT control_runner_dev
    FILES_MATCHING PATTERN "*.h"
)

# Generate and install the package configuration file (ControlRunnerConfig.cmake)
# This allows `find_package(control_runner)` to work.

# Configure the package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ControlRunnerConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/ControlRunnerConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/control_runner/cmake"
    PATH_VARS CMAKE_INSTALL_PREFIX CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
)

# Generate the version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/ControlRunnerConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Install the CMake configuration files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/ControlRunnerConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/ControlRunnerConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/control_runner/cmake"
    COMPONENT control_runner_dev
)

# --- CPack configuration (Optional, for creating packages like .zip, .tar.gz, .deb, .rpm) ---
# set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
# set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
# # ... other CPack settings ...
# include(CPack)

#===================================================================================================

# # control_runner/CMakeLists.txt (when used with Top-Level CMake)
# cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# project(control_runner CXX)

# # --- Define Library Sources ---
# set(CONTROL_RUNNER_SOURCES src/control_runner.cpp)
# set(CONTROL_RUNNER_PUBLIC_HEADERS include/control_runner.h)

# # --- Define Library Target ---
# # Inherits BUILD_SHARED_LIBS from parent scope
# add_library(control_runner SHARED ${CONTROL_RUNNER_SOURCES} ${CONTROL_RUNNER_PUBLIC_HEADERS})
# add_library(MonkeyComms::ControlRunner ALIAS control_runner)

# # --- Configure Include Directories ---
# # PUBLIC includes point to this library's include dir
# # PRIVATE includes for src dir
# # ET includes come from linked targets automatically
# target_include_directories(
#     control_runner
#     PUBLIC
#         $<INSTALL_INTERFACE:include>
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/src
# )

# # --- Link Dependencies ---
# # Link against targets built by the parent scope's add_subdirectory(executorch)
# target_link_libraries(
#     control_runner
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
#     TARGETS control_runner
#     EXPORT control_runner-targets
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#     INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )

# # Install the headers
# install(
#     FILES ${CONTROL_RUNNER_PUBLIC_HEADERS}
#     DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )

# # Generate and install CMake package config files

# # Version file
# write_basic_package_version_file(
#     "control_runner-config_version.cmake"
#     VERSION 1.0.0
#     COMPATIBILITY SameMajorVersion
# )

# # Config file
# configure_package_config_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/control_runner-config.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/control_runner-config.cmake"
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/control_runner
# )

# # Install the config files
# install(
#     FILES
#         "${CMAKE_CURRENT_BINARY_DIR}/control_runner-config.cmake"
#         "${CMAKE_CURRENT_BINARY_DIR}/control_runner-config_version.cmake"
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/control_runner
# )

# # Install the export file
# install(
#     EXPORT control_runner-targets
#     FILE control_runner-targets.cmake
#     NAMESPACE MonkeyComms::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/control_runner
# )
