# TopLevelProject/CMakeLists.txt
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(MyProject CXX)

# --- Set a default install prefix to a user-writable location if not specified ---
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Default installation path" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include GNUInstallDirs for standard installation directories
include(GNUInstallDirs)

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
add_subdirectory(executorch) # Builds ET *once*

# --- Configure ControlRunner and RemoteRunner ---
option(BUILD_CONTROL_RUNNER "Build ControlRunner" ON)
option(BUILD_REMOTE_RUNNER "Build RemoteRunner" ON)

# --- Add subdirectory for ControlRunner ---
if(BUILD_CONTROL_RUNNER)
  message(STATUS "Adding ControlRunner subdirectory...")
  add_subdirectory(ControlRunner)
endif()

# --- Add subdirectory for RemoteRunner ---
if(BUILD_REMOTE_RUNNER)
  message(STATUS "Adding RemoteRunner subdirectory...")
  add_subdirectory(RemoteRunner)
endif()

# --- Configure installation ---
set(INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}" CACHE PATH "Installation directory for header files")