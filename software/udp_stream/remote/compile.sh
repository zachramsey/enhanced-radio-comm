#!/bin/bash

# Compile the UDP video server
g++ -std=c++17 udp_video_server.cpp -o udp_video_server `pkg-config --cflags --libs opencv4` -pthread

# Make the binary executable
chmod +x udp_video_server

echo "Compilation complete. Run with: ./udp_video_server"
echo "Optional parameters:"
echo "  --port PORT       Port to listen on (default: 8888)"
echo "  --camera INDEX    Camera device index (default: 0)"
echo "  --width WIDTH     Frame width (default: 640)"
echo "  --height HEIGHT   Frame height (default: 480)"
echo "  --fps FPS         Target frames per second (default: 30)"
echo "  --client IP       Client IP address (if not specified, waits for first packet)"
