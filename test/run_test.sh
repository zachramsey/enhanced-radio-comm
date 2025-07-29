#!/bin/bash

# Exit on error
set -e

# Configure compiler settings
CXX="g++"
CXXFLAGS="-std=c++17 -Wall -Wextra -O2"

# Define project root directory
PROJECT_ROOT=$(dirname "$(realpath "$0")")/..

# # Define include directories
# INCLUDE_DIRS=(
#     "-I$PROJECT_ROOT/control_runner/include"
#     "-I$PROJECT_ROOT/remote_runner/include"
#     "-Iinclude"
# )

# # Define library directories
# LIBRARY_DIRS=(
#     "-L$PROJECT_ROOT/control_runner/lib"
#     "-L$PROJECT_ROOT/remote_runner/lib"
# )

# Define include directories
INCLUDE_DIRS=(
    "-I$PROJECT_ROOT/monkeycomms/include"
    "-Iinclude"
)

# Define library directories
LIBRARY_DIRS=(
    "-L$PROJECT_ROOT/monkeycomms/lib"
)

# Define libraries to link
LIBRARIES=(
    "-lcontrol_runner"
    "-lremote_runner"
)

# Define source files
SOURCES=(
    "src/libs_test.cpp"
    "src/lodepng.cpp"
)

# Define output binary
OUTPUT="libs_test"

# Print build information
echo "Building $OUTPUT..."
echo "Compiler: $CXX"
echo "Compiler flags: $CXXFLAGS"
echo "Include directories: ${INCLUDE_DIRS[*]}"
echo "Library directories: ${LIBRARY_DIRS[*]}"
echo "Libraries: ${LIBRARIES[*]}"
echo "Source files: ${SOURCES[*]}"

# Execute the compilation and linking
COMPILE_CMD="$CXX $CXXFLAGS ${INCLUDE_DIRS[*]} ${SOURCES[*]} ${LIBRARY_DIRS[*]} ${LIBRARIES[*]} -o $OUTPUT"
echo "Executing: $COMPILE_CMD"
$COMPILE_CMD

# export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/zramsey/Documents/Academic/UI/25_Sp/Sr_Design/Project/control_runner/lib:/home/zramsey/Documents/Academic/UI/25_Sp/Sr_Design/Project/remote_runner/lib

export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/zramsey/Documents/Academic/UI/25_Sp/Sr_Design/Project/monkeycomms/lib


# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Build successful. Binary created: $OUTPUT"
    
    # Set up runtime library path
    echo ""
    echo "To run the compiled binary, you may need to set LD_LIBRARY_PATH:"
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:./control_runner/lib:./remote_runner/lib"
    echo ""
else
    echo "Build failed."
    exit 1
fi

echo "Done."

