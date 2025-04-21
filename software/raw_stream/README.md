# Raw BGR Video Streaming

This directory contains code for streaming raw BGR video frames from an Orange Pi (remote device) to a laptop (control device) without any compression.

## Directory Structure

- `remote/`: Code for the Orange Pi (server)
- `control/`: Code for your laptop (client)

## Setup Instructions

### Remote Device (Orange Pi with DietPi OS)

1. **Install OpenCV and dependencies**:

```bash
# Update package lists
sudo apt-get update

# Install build tools and dependencies
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran

# Install OpenCV
sudo apt-get install -y libopencv-dev
```

2. **Verify OpenCV installation**:

```bash
pkg-config --modversion opencv4
```

3. **Copy the remote directory to your Orange Pi**:

```bash
# From your laptop
scp -r software/raw_stream/remote/ dietpi@dietpi.local:~/raw_stream/
```

4. **Compile the server on the Orange Pi**:

```bash
# SSH into your Orange Pi
ssh dietpi@dietpi.local

# Navigate to the directory and compile
cd ~/raw_stream
chmod +x compile.sh
./compile.sh
```

5. **Run the server on the Orange Pi**:

```bash
# Basic usage (uses default camera at /dev/video0)
./raw_video_server

# With custom parameters
./raw_video_server --camera 1 --fps 30
```

### Control Device (Your Laptop)

#### Option 1: Native C++ Client

1. **Install OpenCV on your laptop**:

For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev
```

For macOS:
```bash
brew install opencv
```

2. **Compile the client**:

```bash
cd software/raw_stream/control
chmod +x compile.sh
./compile.sh
```

3. **Run the client**:

```bash
# Replace with your Orange Pi's IP address or hostname
./raw_video_client --ip xxx.xxx.x.xx
```

#### Option 2: Web-Based Client

still under development

## Usage Notes

### Camera Selection

On the Orange Pi, you may need to specify the correct camera device:

```bash
# List available video devices
ls -l /dev/video*

# Run with specific camera
./raw_video_server --camera 1 --fps 30 # Use /dev/video1
```

### Troubleshooting

1. **Camera not found**:
   - Verify the camera is connected and recognized: `v4l2-ctl --list-devices`
   - Try different camera indices: `--camera 0`, `--camera 1`, etc.

2. **Connection issues**:
   - Ensure both devices are on the same network
   - Check firewall settings to allow traffic on port 8888
   - Verify the correct IP address or hostname is used

3. **Compilation errors**:
   - Ensure OpenCV is properly installed: `pkg-config --modversion opencv4`
   - Check for missing dependencies: `pkg-config --libs opencv4`

