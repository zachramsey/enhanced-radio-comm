#!/bin/bash

# Compile the raw video client
g++ -std=c++11 raw_video_client.cpp -o raw_video_client `pkg-config --cflags --libs opencv4`

# Make the binary executable
chmod +x raw_video_client

echo "Compilation complete. Run with: ./raw_video_client --ip <orange_pi_ip>"
echo "Optional parameters:"
echo "  --ip IP           Server IP address (default: 127.0.0.1)"
echo "  --port PORT       Server port (default: 8888)"
