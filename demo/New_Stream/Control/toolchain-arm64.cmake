# toolchain-arm64.cmake
#############################################
# Cross-compiling to ARM 64-bit (aarch64)
#############################################

# 1. Target system
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 2. Cross-compilation environment
set(CMAKE_CROSSCOMPILING TRUE)

# 3. Sysroot configuration
set(SYSROOT ${CMAKE_SOURCE_DIR}/third_party/python3.11-dev-arm64/usr CACHE PATH "")
set(CMAKE_SYSROOT ${SYSROOT})

# Tell the compiler to use only sysroot includes
set(CMAKE_C_FLAGS   "--sysroot=${SYSROOT} -nostdinc"       CACHE STRING "")
set(CMAKE_CXX_FLAGS "--sysroot=${SYSROOT} -nostdinc++"     CACHE STRING "")

# include directories for libstdc++ and libc6-dev
# you may need to adjust the 12 to your version of libstdc++
include_directories(
  ${SYSROOT}/include
  ${SYSROOT}/include/aarch64-linux-gnu
  ${CMAKE_SOURCE_DIR}/third_party/libstdc++-arm64/usr/include/c++/12
  ${CMAKE_SOURCE_DIR}/third_party/libstdc++-arm64/usr/include/aarch64-linux-gnu/c++/12
  ${CMAKE_SOURCE_DIR}/third_party/libc6-dev-arm64/usr/include
  ${CMAKE_SOURCE_DIR}/third_party/libc6-dev-arm64/usr/include/aarch64-linux-gnu
)

# Also pass sysroot to linker flags just in case
set(CMAKE_EXE_LINKER_FLAGS "--sysroot=${SYSROOT}" CACHE STRING "")
set(CMAKE_MODULE_LINKER_FLAGS "--sysroot=${SYSROOT}" CACHE STRING "")
set(CMAKE_SHARED_LINKER_FLAGS "--sysroot=${SYSROOT}" CACHE STRING "")


# 4. Cross-compilers (full paths to ARM64 toolchain)
set(CMAKE_C_COMPILER   /usr/bin/aarch64-linux-gnu-gcc CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++ CACHE FILEPATH "")

# 5. Python dev paths (from extracted python3.11-dev and libpython3.11)
set(Python3_INCLUDE_DIRS ${SYSROOT}/include/aarch64-linux-gnu/python3.11 CACHE PATH "")
set(Python3_LIBRARIES    ${SYSROOT}/lib/aarch64-linux-gnu/libpython3.11.so CACHE FILEPATH "")

# 6. Include directories for additional headers
include_directories(${SYSROOT}/include)
include_directories(${SYSROOT}/include/aarch64-linux-gnu)

# 7. Find path behavior (search in sysroot first)
set(CMAKE_FIND_ROOT_PATH ${SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# 8. Pybind11 configuration overrides
set(PYBIND11_FINDPYTHON New CACHE STRING "")
set(PYTHON_MODULE_EXTENSION ".cpython-311-aarch64-linux-gnu.so" CACHE STRING "")
