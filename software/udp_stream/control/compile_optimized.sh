#!/bin/bash

# Compile the optimized UDP video client
g++ -std=c++17 optimized_udp_client.cpp -o optimized_udp_client `pkg-config --cflags --libs opencv4` -pthread -O3

# Make the binary executable
chmod +x optimized_udp_client

echo "Compilation complete. Run with: ./optimized_udp_client --server SERVER_IP"
echo "Optional parameters:"
echo "  --port PORT       Port to listen on (default: 8888)"
echo "  --buffer SIZE     Frame buffer size (default: 5)"
echo "  --display-width W Display width (default: 320)"
echo "  --display-height H Display height (default: 240)"
echo "  --frame-skip N    Display every Nth frame (default: 1)"
echo "  --no-display      Don't display frames (for headless operation)"
