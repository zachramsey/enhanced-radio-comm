#!/usr/bin/env python3

import os
import json
import time
import socket
import logging
import threading
import webbrowser
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from flask import Flask, render_template, jsonify, request
import zeroconf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("control_device.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("control_device")

# Initialize Flask app
app = Flask(__name__)

# Global variables
remote_device = None
data_usage = {
    "total_bytes_received": 0,
    "current_bitrate": 0,
    "start_time": time.time()
}
config_file = Path("config/control_config.json")

class RemoteDeviceDiscovery:
    """
    Handles discovery of remote devices using mDNS/Zeroconf
    """
    def __init__(self):
        self.zeroconf = zeroconf.Zeroconf()
        self.browser = None
        self.devices = {}
        self.lock = threading.Lock()

    def start_discovery(self):
        """Start discovering remote devices"""
        logger.info("Starting remote device discovery")
        self.browser = zeroconf.ServiceBrowser(
            self.zeroconf,
            "_enhanced-radio._tcp.local.",
            self
        )

    def stop_discovery(self):
        """Stop discovering remote devices"""
        if self.browser:
            self.browser.cancel()
        self.zeroconf.close()

    def add_service(self, zc, type, name):
        """Called when a new service is discovered"""
        info = zc.get_service_info(type, name)
        if info:
            with self.lock:
                self.devices[name] = {
                    "name": name.split(".")[0],
                    "address": socket.inet_ntoa(info.addresses[0]),
                    "port": info.port,
                    "properties": {k.decode(): v.decode() for k, v in info.properties.items()}
                }
                logger.info(f"Found remote device: {self.devices[name]}")

    def remove_service(self, zc, type, name):
        """Called when a service is removed"""
        with self.lock:
            if name in self.devices:
                logger.info(f"Remote device lost: {name}")
                del self.devices[name]

    def get_devices(self) -> Dict:
        """Return discovered devices"""
        with self.lock:
            return self.devices.copy()

def load_config() -> Dict:
    """Load configuration from file"""
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return {
        "role": "control",
        "auto_launch": True,
        "preferred_remote": None,
        "port": 5000
    }

def save_config(config: Dict):
    """Save configuration to file"""
    config_file.parent.mkdir(exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

def update_data_usage(bytes_received: int):
    """Update data usage statistics"""
    global data_usage

    current_time = time.time()
    elapsed = current_time - data_usage["start_time"]

    data_usage["total_bytes_received"] += bytes_received

    # Calculate bitrate over the last second
    if elapsed >= 1.0:
        data_usage["current_bitrate"] = (data_usage["total_bytes_received"] * 8) / elapsed
        data_usage["start_time"] = current_time
        data_usage["total_bytes_received"] = 0

def auto_launch_browser():
    """Auto-launch the browser with the web interface"""
    config = load_config()
    if config.get("auto_launch", True):
        time.sleep(1)  # Give the server a moment to start
        url = f"http://localhost:{config.get('port', 5000)}"
        logger.info(f"Auto-launching browser at {url}")
        webbrowser.open(url)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/devices')
def get_devices():
    """API endpoint to get discovered devices"""
    global discovery
    return jsonify(list(discovery.get_devices().values()))

@app.route('/api/connect', methods=['POST'])
def connect_to_device():
    """API endpoint to connect to a specific device"""
    global remote_device

    device_address = request.json.get('address')
    device_port = request.json.get('port')

    if not device_address or not device_port:
        return jsonify({"success": False, "error": "Missing address or port"}), 400

    # Here you would establish the connection to the remote device
    # For now, we'll just store the information
    remote_device = {
        "address": device_address,
        "port": device_port,
        "connected_at": time.time()
    }

    # Update configuration with preferred remote
    config = load_config()
    config["preferred_remote"] = device_address
    save_config(config)

    return jsonify({"success": True})

@app.route('/api/stats')
def get_stats():
    """API endpoint to get data usage statistics"""
    global data_usage
    return jsonify(data_usage)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """API endpoint to update user settings"""
    if not request.json:
        return jsonify({"success": False, "error": "No data provided"}), 400

    config = load_config()

    # Update auto_launch setting if provided
    if 'auto_launch' in request.json:
        config['auto_launch'] = request.json['auto_launch']

    save_config(config)
    return jsonify({"success": True})

def main():
    """Main entry point"""
    global discovery

    # Load configuration
    config = load_config()

    # Start device discovery
    discovery = RemoteDeviceDiscovery()
    discovery.start_discovery()

    # Start browser auto-launch in a separate thread
    if config.get("auto_launch", True):
        threading.Thread(target=auto_launch_browser, daemon=True).start()

    # Start the Flask app
    app.run(
        host='0.0.0.0',
        port=config.get('port', 5000),
        debug=False  # Set to False in production
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
        if 'discovery' in globals():
            discovery.stop_discovery()
