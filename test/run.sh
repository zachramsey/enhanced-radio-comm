#!/bin/bash
# filepath: /home/zramsey/Documents/Academic/UI/25_Sp/Sr_Design/Project/test/run.sh

# Exit on error
set -e

# Directory paths
MONKEY_COMMS_DIR="../libmonkeycomms"
INCLUDE_DIR="$MONKEY_COMMS_DIR/include"
LIB_DIR="$MONKEY_COMMS_DIR/lib"

# Output binary name
OUTPUT_BIN="libs_test_bin"

echo "Compiling libs_test.cpp..."

# Compile the test file with all necessary include paths
g++ -o $OUTPUT_BIN libs_test.cpp \
  -I$INCLUDE_DIR \
  -L$LIB_DIR \
  -lcontrol_runner \
  -lremote_runner \
  -lpng \
  -lm \
  -std=c++17 \
  -Wl,-rpath,$LIB_DIR

echo "Setting up library paths..."
# Make sure runtime can find the required libraries
export LD_LIBRARY_PATH=$LIB_DIR:$LD_LIBRARY_PATH

echo "Running the test program..."
gdb $OUTPUT_BIN

echo "Test completed!"
