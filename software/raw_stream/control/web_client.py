#!/usr/bin/env python3
"""
Web-based client for raw video streaming
This script connects to the raw video server on the Orange Pi,
receives the raw BGR frames, and serves them as MJPEG over HTTP.
"""

import cv2
import numpy as np
import socket
import struct
import threading
import time
import argparse
from flask import Flask, Response, render_template
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_client")

# Frame header structure (must match C++ server)
HEADER_FORMAT = "iiii"  # width, height, channels, dataSize
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Global variables
latest_frame = None
frame_lock = threading.Lock()
connection_status = "Disconnected"
stats = {
    "fps": 0,
    "data_rate_mbps": 0,
    "resolution": "N/A",
    "total_frames": 0
}

# Initialize Flask app
app = Flask(__name__)

def receive_all(sock, n):
    """Receive exactly n bytes from socket"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def video_receiver_thread(server_ip, server_port):
    """Thread to receive video frames from the server"""
    global latest_frame, connection_status, stats
    
    while True:
        try:
            # Create socket and connect to server
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((server_ip, server_port))
            
            connection_status = "Connected"
            logger.info(f"Connected to server at {server_ip}:{server_port}")
            
            # Variables for FPS and data rate calculation
            frame_count = 0
            start_time = time.time()
            total_bytes = 0
            data_rate_start = time.time()
            
            while True:
                # Receive header
                header_data = receive_all(client_socket, HEADER_SIZE)
                if header_data is None:
                    logger.error("Failed to receive header")
                    break
                
                width, height, channels, data_size = struct.unpack(HEADER_FORMAT, header_data)
                
                # Validate header
                if width <= 0 or height <= 0 or channels <= 0 or data_size <= 0:
                    logger.error(f"Invalid header: {width}x{height}, {channels} channels, {data_size} bytes")
                    break
                
                # Receive frame data
                frame_data = receive_all(client_socket, data_size)
                if frame_data is None:
                    logger.error("Failed to receive frame data")
                    break
                
                # Update data rate statistics
                total_bytes += HEADER_SIZE + data_size
                now = time.time()
                data_rate_elapsed = now - data_rate_start
                if data_rate_elapsed >= 1:
                    stats["data_rate_mbps"] = (total_bytes * 8 / 1000000) / data_rate_elapsed
                    total_bytes = 0
                    data_rate_start = now
                
                # Convert to numpy array and create OpenCV Mat
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, channels)
                
                # Update FPS counter
                frame_count += 1
                stats["total_frames"] += 1
                fps_elapsed = now - start_time
                if fps_elapsed >= 1:
                    stats["fps"] = frame_count / fps_elapsed
                    frame_count = 0
                    start_time = now
                
                # Update resolution info
                stats["resolution"] = f"{width}x{height}"
                
                # Add stats to the frame
                stats_text = f"Resolution: {stats['resolution']} | FPS: {stats['fps']:.1f} | Data Rate: {stats['data_rate_mbps']:.2f} Mbps"
                cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update the latest frame (thread-safe)
                with frame_lock:
                    latest_frame = frame.copy()
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            connection_status = "Disconnected"
            time.sleep(2)  # Wait before reconnecting
        finally:
            try:
                client_socket.close()
            except:
                pass

def generate_mjpeg():
    """Generate MJPEG stream from the latest frame"""
    global latest_frame
    
    while True:
        # Wait for a frame to be available
        if latest_frame is None:
            time.sleep(0.1)
            continue
        
        # Get the latest frame (thread-safe)
        with frame_lock:
            frame = latest_frame.copy()
        
        # Convert BGR to JPEG
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Yield the JPEG frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        # Control the frame rate of the MJPEG stream
        time.sleep(0.03)  # ~30 FPS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for the video feed"""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Route for status information"""
    return {
        "connection": connection_status,
        "stats": stats
    }

def create_template_files():
    """Create the necessary template files"""
    import os
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html
    with open('templates/index.html', 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raw Video Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .video-container {
            margin: 20px auto;
            text-align: center;
        }
        .video-feed {
            max-width: 100%;
            border: 2px solid #333;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .status-panel {
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .status-item {
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
        }
        .status-item:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        .status-label {
            font-weight: bold;
            color: #555;
        }
        .status-value {
            color: #333;
        }
        .connected {
            color: green;
        }
        .disconnected {
            color: red;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Raw Video Stream</h1>
        
        <div class="video-container">
            <img src="/video_feed" class="video-feed" alt="Video Stream">
        </div>
        
        <div class="status-panel">
            <div class="status-item">
                <span class="status-label">Connection Status:</span>
                <span id="connection-status" class="status-value disconnected">Disconnected</span>
            </div>
            <div class="status-item">
                <span class="status-label">Resolution:</span>
                <span id="resolution" class="status-value">N/A</span>
            </div>
            <div class="status-item">
                <span class="status-label">FPS:</span>
                <span id="fps" class="status-value">0</span>
            </div>
            <div class="status-item">
                <span class="status-label">Data Rate:</span>
                <span id="data-rate" class="status-value">0 Mbps</span>
            </div>
            <div class="status-item">
                <span class="status-label">Total Frames:</span>
                <span id="total-frames" class="status-value">0</span>
            </div>
        </div>
    </div>
    
    <script>
        // Update status information periodically
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    // Update connection status
                    const connectionStatus = document.getElementById('connection-status');
                    connectionStatus.textContent = data.connection;
                    connectionStatus.className = 'status-value ' + data.connection.toLowerCase();
                    
                    // Update stats
                    document.getElementById('resolution').textContent = data.stats.resolution;
                    document.getElementById('fps').textContent = data.stats.fps.toFixed(1);
                    document.getElementById('data-rate').textContent = data.stats.data_rate_mbps.toFixed(2) + ' Mbps';
                    document.getElementById('total-frames').textContent = data.stats.total_frames;
                })
                .catch(error => console.error('Error fetching status:', error));
        }
        
        // Update status every second
        setInterval(updateStatus, 1000);
        
        // Initial update
        updateStatus();
    </script>
</body>
</html>""")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Web client for raw video streaming')
    parser.add_argument('--ip', default='127.0.0.1', help='Server IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8888, help='Server port (default: 8888)')
    parser.add_argument('--web-port', type=int, default=5000, help='Web server port (default: 5000)')
    args = parser.parse_args()
    
    # Create template files
    create_template_files()
    
    # Start video receiver thread
    receiver_thread = threading.Thread(
        target=video_receiver_thread,
        args=(args.ip, args.port),
        daemon=True
    )
    receiver_thread.start()
    
    # Start Flask app
    logger.info(f"Starting web server on port {args.web_port}")
    app.run(host='0.0.0.0', port=args.web_port, threaded=True)
