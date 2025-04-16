#!/bin/bash

# Compile the raw video server
g++ -std=c++11 raw_video_server.cpp -o raw_video_server `pkg-config --cflags --libs opencv4`

# Make the binary executable
chmod +x raw_video_server

echo "Compilation complete. Run with: ./raw_video_server"
echo "Optional parameters:"
echo "  --port PORT       Port to listen on (default: 8888)"
echo "  --camera INDEX    Camera device index (default: 0)"
echo "  --width WIDTH     Frame width (default: 640)"
echo "  --height HEIGHT   Frame height (default: 480)"
echo "  --fps FPS         Target frames per second (default: 30)"
