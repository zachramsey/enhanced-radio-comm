# UDP Video Streaming

This directory contains code for streaming raw BGR888 video frames from an Orange Pi (server) to a laptop (client) using UDP protocol.

## Features

- Raw BGR888 video streaming over UDP
- Web-based client interface
- Real-time statistics (FPS, data rate, etc.)
- Configurable buffer size and port
- Automatic frame reassembly from UDP packets

## Directory Structure

- `remote/`: Code for the Orange Pi (server)
  - `udp_video_server.cpp`: Main server code
  - `compile.sh`: Compilation script

- `control/`: Code for your laptop (client)
  - `udp_video_client.cpp`: C++ client for receiving UDP video
  - `compile.sh`: Compilation script for C++ client
  - `web_server.py`: Python Flask server for web interface
  - `templates/`: HTML templates
  - `static/`: CSS, JavaScript, and other static files

## Setup Instructions

### Remote Device (Orange Pi with DietPi OS)

1. **Install OpenCV and dependencies**:

```bash
# Update package lists
sudo apt-get update

# Install build tools and dependencies
sudo apt-get install -y build-essential cmake git pkg-config

# Install image and video I/O libraries
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev

# Install GTK for GUI features (optional)
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev

# Install optimization libraries
sudo apt-get install -y libatlas-base-dev gfortran

# Install Python development headers (if you need Python bindings)
sudo apt-get install -y python3-dev

# Install OpenCV
sudo apt-get install -y libopencv-dev
```

2. **Compile the UDP video server**:

```bash
cd remote
chmod +x compile.sh
./compile.sh
```

3. **Run the server**:

```bash
# Basic usage (will wait for a client to connect)
./udp_video_server

# Specify client IP address
./udp_video_server --client 192.168.1.100

# Specify camera device, resolution, and FPS
./udp_video_server --camera 0 --width 640 --height 480 --fps 30
```

### Control Device (Laptop)

1. **Install OpenCV and dependencies**:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config
sudo apt-get install -y libopencv-dev python3-opencv
sudo apt-get install -y python3-pip python3-flask python3-numpy

# Install Flask-SocketIO
pip3 install flask-socketio
```

2. **Compile the C++ client**:

```bash
cd control
chmod +x compile.sh
./compile.sh
```

3. **Run the web server**:

```bash
cd control
python3 web_server.py
```

4. **Access the web interface**:

Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Web Interface

1. Enter the IP address of your Orange Pi in the "Server IP" field
2. Click "Save Configuration"
3. Click "Start Stream" to begin receiving video
4. Adjust buffer size if needed
5. Click "Stop Stream" to stop the video stream

### Direct C++ Client (without web interface)

You can also run the C++ client directly without the web interface:

```bash
cd control
./udp_video_client --server 192.168.1.101
```

## Troubleshooting

- **No video appears**: Make sure the server is running and the IP address is correct
- **Video is laggy**: Try reducing the resolution or FPS on the server
- **Packet loss**: Increase the buffer size in the web interface
- **Camera not found**: Check the camera device index (--camera parameter)

## Technical Details

### Protocol

The UDP video streaming protocol works as follows:

1. Client sends an initial packet to the server to establish connection
2. Server captures frames from the camera
3. For each frame:
   - Server sends a frame header packet with dimensions and frame number
   - Server splits the frame data into multiple packets and sends them
4. Client reassembles the packets into complete frames
5. Client displays the frames in the web interface

### Frame Format

- Raw BGR888 format (3 bytes per pixel)
- No compression is applied during transmission
- Optional H.264 conversion can be implemented on the client side

## License

This project is open source and available under the MIT License.
