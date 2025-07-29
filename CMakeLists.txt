
cmake_minimum_required(VERSION 3.15) # Executorch might require a newer version, adjust if necessary
project(MonkeyComms VERSION 1.0.0 DESCRIPTION "Control and Remote Runners with Executorch" LANGUAGES CXX)

# Standard C++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set debugging option
option(DEBUG_MODE "Enable debug mode" OFF)
if(DEBUG_MODE)
    message(STATUS "Debug mode is enabled.")
    set(CMAKE_BUILD_TYPE Debug)
    set(EXECUTORCH_LOG_LEVEL Debug)
    set(EXECUTORCH_BUILD_DEVTOOLS ON)
    set(EXECUTORCH_ENABLE_EVENT_TRACER ON)
else()
    message(STATUS "Debug mode is disabled.")
    set(CMAKE_BUILD_TYPE Release)
endif()

# Configure ExecuTorch Build
option(OPTIMIZE_SIZE "Optimize size" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "Build the Module extension" ON)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "Build the Tensor extension" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "Build optimized kernels" ON)
option(EXECUTORCH_BUILD_KERNELS_QUANTIZED "Build quantized kernels" ON)
option(EXECUTORCH_BUILD_XNNPACK "Build the XNNPACK backend" ON)

# Add the source directory
add_subdirectory(src)

# --- Installation ---
# This section is important if you want to install all components (executorch, control_runner, remote_runner)
# from the top level. Individual components also have their install rules.

# Include GNUInstallDirs for standard directory names (e.g., lib, include, bin)
include(GNUInstallDirs)

# Example: Setting a global install prefix if not specified by the user/environment
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/monkeycomms" CACHE PATH "Default install prefix for MonkeyComms" FORCE)
endif()

message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")

#===================================================================================================

# # src/CMakeLists.txt
# cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
# project(monkeycomms CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Include GNUInstallDirs for standard installation directories
# include(GNUInstallDirs)

# # Include CMake's package config helpers
# include(CMakePackageConfigHelpers)

# # --- Configure ControlRunner and RemoteRunner ---
# option(BUILD_CONTROL_RUNNER "Build control_runner" OFF)
# option(BUILD_REMOTE_RUNNER "Build remote_runner" OFF)

# # --- Set a default install prefix to a user-writable location if not specified ---
# if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#   if(BUILD_CONTROL_RUNNER AND NOT BUILD_REMOTE_RUNNER)
#     set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/control_runner" CACHE PATH "Default install prefix for control_runner" FORCE)
#   elseif(BUILD_REMOTE_RUNNER AND NOT BUILD_CONTROL_RUNNER)
#     set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/remote_runner" CACHE PATH "Default install prefix for remote_runner" FORCE)
#   else()
#     set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/monkeycomms" CACHE PATH "Default install prefix for MonkeyComms" FORCE)
#   endif()
# endif()

# # --- Set debugging mode ---
# option(DEBUG_MODE "Enable debug mode" OFF)
# if(DEBUG_MODE)
#   message(STATUS "Debug mode is enabled.")
#   set(CMAKE_BUILD_TYPE Debug)
#   set(EXECUTORCH_LOG_LEVEL Debug)
#   set(EXECUTORCH_BUILD_DEVTOOLS ON)
#   set(EXECUTORCH_ENABLE_EVENT_TRACER ON)
# else()
#   message(STATUS "Debug mode is disabled.")
#   set(CMAKE_BUILD_TYPE Release)
# endif()

# # --- Configure ExecuTorch Build ---
# # option(OPTIMIZE_SIZE "Optimize size" ON)
# option(EXECUTORCH_BUILD_EXTENSION_MODULE "Build the Module extension" ON)
# option(EXECUTORCH_BUILD_EXTENSION_TENSOR "Build the Tensor extension" ON)
# option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "Build optimized kernels" ON)
# option(EXECUTORCH_BUILD_KERNELS_QUANTIZED "Build quantized kernels" ON)
# option(EXECUTORCH_BUILD_XNNPACK "Build the XNNPACK backend" ON)

# # --- Configure CMake for ExecuTorch ---
# message(STATUS "Adding executorch subdirectory...")
# add_subdirectory(src/executorch) # Builds ET *once*

# # --- Add subdirectory for ControlRunner ---
# if(BUILD_CONTROL_RUNNER OR NOT BUILD_REMOTE_RUNNER)
#   message(STATUS "Adding control_runner subdirectory...")
#   add_subdirectory(src/control_runner)
# endif()

# # --- Add subdirectory for RemoteRunner ---
# if(BUILD_REMOTE_RUNNER OR NOT BUILD_CONTROL_RUNNER)
#   message(STATUS "Adding remote_runner subdirectory...")
#   add_subdirectory(src/remote_runner)
# endif()
