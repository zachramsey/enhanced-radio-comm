#!/usr/bin/env python3

import os
import sys
import json
import time
import socket
import logging
import threading
import subprocess
import webbrowser
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("udp_stream.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("udp_stream")

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
config = {
    "port": 5000,
    "udp_port": 8888,
    "auto_launch": True,
    "server_ip": "",
    "buffer_size": 10
}

# Video streaming globals
client_process = None
streaming_active = False
frame_buffer = None
latest_frame = None
frame_lock = threading.Lock()
stats = {
    "fps": 0,
    "data_rate": 0,
    "resolution": "0x0",
    "total_bytes": 0
}

def load_config():
    """Load configuration from file"""
    config_path = Path("config.json")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                logger.info(f"Loaded configuration: {config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

def save_config():
    """Save configuration to file"""
    config_path = Path("config.json")
    try:
        os.makedirs(config_path.parent, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            logger.info(f"Saved configuration: {config}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

def auto_launch_browser():
    """Auto-launch the browser with the web interface"""
    if config.get("auto_launch", True):
        time.sleep(1)  # Give the server a moment to start
        url = f"http://localhost:{config.get('port', 5000)}"
        logger.info(f"Auto-launching browser at {url}")
        webbrowser.open(url)

def start_udp_client():
    """Start the UDP client process"""
    global client_process, streaming_active
    
    if client_process is not None:
        logger.warning("UDP client already running")
        return False
    
    if not config["server_ip"]:
        logger.error("Server IP not configured")
        return False
    
    try:
        # Build command
        cmd = [
            "./udp_video_client",
            "--server", config["server_ip"],
            "--port", str(config["udp_port"]),
            "--buffer", str(config["buffer_size"])
        ]
        
        # Start process
        logger.info(f"Starting UDP client: {' '.join(cmd)}")
        client_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Start thread to read output
        threading.Thread(target=read_client_output, daemon=True).start()
        
        streaming_active = True
        return True
    except Exception as e:
        logger.error(f"Error starting UDP client: {e}")
        return False

def stop_udp_client():
    """Stop the UDP client process"""
    global client_process, streaming_active
    
    if client_process is None:
        logger.warning("UDP client not running")
        return True
    
    try:
        logger.info("Stopping UDP client")
        client_process.terminate()
        client_process.wait(timeout=5)
        client_process = None
        streaming_active = False
        return True
    except Exception as e:
        logger.error(f"Error stopping UDP client: {e}")
        try:
            client_process.kill()
            client_process = None
        except:
            pass
        streaming_active = False
        return False

def read_client_output():
    """Read and process output from the UDP client"""
    global client_process, stats
    
    if client_process is None:
        return
    
    for line in client_process.stdout:
        line = line.strip()
        logger.debug(f"UDP client: {line}")
        
        # Parse statistics from output
        if "FPS:" in line and "Data Rate:" in line:
            try:
                # Extract FPS
                fps_start = line.find("FPS:") + 4
                fps_end = line.find("|", fps_start)
                fps = float(line[fps_start:fps_end].strip())
                
                # Extract data rate
                rate_start = line.find("Data Rate:") + 10
                rate_end = line.find("Mbps", rate_start)
                data_rate = float(line[rate_start:rate_end].strip())
                
                # Extract resolution
                res_start = line.find("Resolution:") + 11
                res_end = line.find("|", res_start)
                resolution = line[res_start:res_end].strip()
                
                # Update stats
                stats["fps"] = fps
                stats["data_rate"] = data_rate
                stats["resolution"] = resolution
                
                # Emit stats to clients
                socketio.emit("stats_update", stats)
            except Exception as e:
                logger.error(f"Error parsing stats: {e}")
    
    # Process has ended
    if client_process is not None:
        stderr = client_process.stderr.read()
        if stderr:
            logger.error(f"UDP client error: {stderr}")
        
        client_process = None
        streaming_active = False
        socketio.emit("stream_stopped")

def generate_frames():
    """Generator function for video streaming"""
    global latest_frame, streaming_active
    
    while True:
        # Wait until we have a frame
        if latest_frame is None:
            if not streaming_active:
                break
            time.sleep(0.01)
            continue
        
        # Convert frame to JPEG
        with frame_lock:
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
            else:
                continue
        
        # Yield the frame in multipart response format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Short sleep to control frame rate
        time.sleep(0.01)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/stats')
def get_stats():
    """API endpoint to get current statistics"""
    return jsonify(stats)

@app.route('/api/config', methods=['GET'])
def get_config():
    """API endpoint to get current configuration"""
    return jsonify(config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """API endpoint to update configuration"""
    try:
        new_config = request.json
        config.update(new_config)
        save_config()
        return jsonify({"success": True, "config": config})
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    socketio.emit("config_update", config, to=request.sid)
    socketio.emit("stats_update", stats, to=request.sid)
    socketio.emit("stream_status", {"active": streaming_active}, to=request.sid)

@socketio.on('start_stream')
def handle_start_stream(data):
    """Handle start stream request"""
    logger.info(f"Start stream request: {data}")
    
    # Update config if server IP is provided
    if "server_ip" in data:
        config["server_ip"] = data["server_ip"]
        save_config()
    
    # Start UDP client
    success = start_udp_client()
    return {"success": success}

@socketio.on('stop_stream')
def handle_stop_stream():
    """Handle stop stream request"""
    logger.info("Stop stream request")
    success = stop_udp_client()
    return {"success": success}

def main():
    """Main entry point"""
    # Load configuration
    load_config()
    
    # Start browser auto-launch in a separate thread
    if config.get("auto_launch", True):
        threading.Thread(target=auto_launch_browser, daemon=True).start()
    
    # Start the Flask app
    socketio.run(
        app,
        host='0.0.0.0',
        port=config.get('port', 5000),
        debug=False,  # Set to False in production
        allow_unsafe_werkzeug=True
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
    finally:
        # Clean up
        if client_process is not None:
            stop_udp_client()
