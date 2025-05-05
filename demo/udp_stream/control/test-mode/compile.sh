#!/bin/bash

# Compile the UDP video client
g++ -std=c++17 udp_video_client.cpp -o udp_video_client `pkg-config --cflags --libs opencv4` -pthread

# Make the binary executable
chmod +x udp_video_client

echo "Compilation complete. Run with: ./udp_video_client --server SERVER_IP"
echo "Optional parameters:"
echo "  --port PORT       Port to listen on (default: 8888)"
echo "  --buffer SIZE     Frame buffer size (default: 10)"
