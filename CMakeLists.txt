# src/CMakeLists.txt
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(MonkeyComms CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include GNUInstallDirs for standard installation directories
include(GNUInstallDirs)

# Include CMake's package config helpers
include(CMakePackageConfigHelpers)

# --- Set a default install prefix to a user-writable location if not specified ---
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/libmonkeycomms" CACHE PATH "Default install prefix" FORCE)
endif()

# --- Set the default build type if not specified ---
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Configure ExecuTorch Build ---
option(OPTIMIZE_SIZE "Optimize size" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "Build the Module extension" ON)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "Build the Tensor extension" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "Build optimized kernels" ON)
option(EXECUTORCH_BUILD_KERNELS_QUANTIZED "Build quantized kernels" ON)
option(EXECUTORCH_BUILD_XNNPACK "Build the XNNPACK backend" ON)

# --- Configure CMake for ExecuTorch ---
message(STATUS "Adding executorch subdirectory...")
add_subdirectory(src/executorch) # Builds ET *once*

# --- Configure ControlRunner and RemoteRunner ---
option(BUILD_CONTROL_RUNNER "Build control_runner" ON)
option(BUILD_REMOTE_RUNNER "Build remote_runner" ON)

# --- Add subdirectory for ControlRunner ---
if(BUILD_CONTROL_RUNNER)
  message(STATUS "Adding control_runner subdirectory...")
  add_subdirectory(src/control_runner)
endif()

# --- Add subdirectory for RemoteRunner ---
if(BUILD_REMOTE_RUNNER)
  message(STATUS "Adding remote_runner subdirectory...")
  add_subdirectory(src/remote_runner)
endif()
