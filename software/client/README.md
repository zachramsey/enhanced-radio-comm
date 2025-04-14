# Enhanced Radio Communication - Control Device

This directory contains the control device (PC/laptop) application for the Enhanced Radio Communication project. The control device discovers remote devices, receives video streams, and displays them in a web interface.

## Features

- **Auto-Discovery**: Automatically finds remote devices on the local network using mDNS
- **Web Interface**: Displays video stream and controls in a browser
- **Data Usage Monitoring**: Tracks and displays bandwidth usage statistics
- **Auto-Launch**: Automatically opens the web interface on startup

## Requirements

- Python 3.6 or higher
- Web browser (Chrome, Firefox, Edge, etc.)
- Network connection to the remote device

## Installation

The application will automatically install required dependencies when launched. The main dependencies are:

- Flask: Web framework for the user interface
- Zeroconf: For mDNS device discovery

## Usage

1. Start the application:
   ```
   python start_control.py
   ```

2. The web interface will automatically open in your default browser.

3. Select a discovered remote device from the list.

4. Click "Start Stream" to begin viewing the video feed.

## Configuration

The application stores its configuration in `config/control_config.json`. You can modify this file to change settings such as:

- `auto_launch`: Whether to automatically open the browser on startup
- `port`: The port to run the web server on
- `preferred_remote`: The last connected remote device

## Troubleshooting

- **No devices found**: Ensure the remote device is powered on and connected to the same network.
- **Cannot connect to device**: Check firewall settings and ensure ports 5000 and 8080 are open.
- **Video stream not showing**: Verify the remote device is properly streaming video.

## Development

The application structure is as follows:

- `app.py`: Main application file
- `start_control.py`: Launcher script
- `templates/`: HTML templates
- `static/`: CSS, JavaScript, and other static files
- `config/`: Configuration files

To modify the web interface, edit the files in the `templates/` and `static/` directories.

