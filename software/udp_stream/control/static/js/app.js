/**
 * UDP Video Streaming - Client JavaScript
 *
 * This script handles:
 * - Server configuration
 * - Video stream control
 * - Statistics display
 * - WebSocket communication
 */

// DOM Elements
const connectionStatus = document.getElementById('connection-status');
const videoPlaceholder = document.getElementById('video-placeholder');
const videoStream = document.getElementById('video-stream');
const serverIpInput = document.getElementById('server-ip');
const udpPortInput = document.getElementById('udp-port');
const saveConfigBtn = document.getElementById('save-config');
const startStreamBtn = document.getElementById('start-stream');
const stopStreamBtn = document.getElementById('stop-stream');
const bufferSizeInput = document.getElementById('buffer-size');
const bufferValue = document.getElementById('buffer-value');
const fpsElement = document.getElementById('fps');
const dataRateElement = document.getElementById('data-rate');
const resolutionElement = document.getElementById('resolution-value');
const bitrateElement = document.getElementById('bitrate');
const resolutionDisplay = document.getElementById('resolution');
const connectionTimeElement = document.getElementById('connection-time');

// Global variables
let socket = null;
let isStreaming = false;
let connectionStartTime = null;
let connectionTimer = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Socket.IO
    socket = io();
    
    // Set up event listeners
    setupEventListeners();
    
    // Set up Socket.IO event handlers
    setupSocketHandlers();
    
    // Update buffer size display
    updateBufferSizeDisplay();
});

// Set up event listeners
function setupEventListeners() {
    saveConfigBtn.addEventListener('click', saveConfiguration);
    startStreamBtn.addEventListener('click', startStream);
    stopStreamBtn.addEventListener('click', stopStream);
    bufferSizeInput.addEventListener('input', updateBufferSizeDisplay);
}

// Set up Socket.IO event handlers
function setupSocketHandlers() {
    socket.on('connect', () => {
        console.log('Connected to server');
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateConnectionStatus('disconnected');
        stopConnectionTimer();
    });
    
    socket.on('config_update', (config) => {
        console.log('Received config update:', config);
        updateConfigDisplay(config);
    });
    
    socket.on('stats_update', (stats) => {
        console.log('Received stats update:', stats);
        updateStatsDisplay(stats);
    });
    
    socket.on('stream_status', (data) => {
        console.log('Received stream status:', data);
        updateStreamStatus(data.active);
    });
    
    socket.on('stream_stopped', () => {
        console.log('Stream stopped');
        updateStreamStatus(false);
    });
}

// Update buffer size display
function updateBufferSizeDisplay() {
    bufferValue.textContent = bufferSizeInput.value;
}

// Save configuration
function saveConfiguration() {
    const config = {
        server_ip: serverIpInput.value,
        udp_port: parseInt(udpPortInput.value),
        buffer_size: parseInt(bufferSizeInput.value)
    };
    
    console.log('Saving configuration:', config);
    
    fetch('/api/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Configuration saved successfully');
            alert('Configuration saved successfully');
        } else {
            console.error('Error saving configuration:', data.error);
            alert('Error saving configuration: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error saving configuration:', error);
        alert('Error saving configuration: ' + error);
    });
}

// Start video stream
function startStream() {
    if (isStreaming) {
        console.log('Stream already active');
        return;
    }
    
    const serverIp = serverIpInput.value;
    if (!serverIp) {
        alert('Please enter a server IP address');
        return;
    }
    
    console.log('Starting stream from server:', serverIp);
    
    socket.emit('start_stream', {
        server_ip: serverIp
    }, (response) => {
        if (response.success) {
            console.log('Stream started successfully');
            updateStreamStatus(true);
            startConnectionTimer();
        } else {
            console.error('Error starting stream');
            alert('Error starting stream');
        }
    });
}

// Stop video stream
function stopStream() {
    if (!isStreaming) {
        console.log('No active stream to stop');
        return;
    }
    
    console.log('Stopping stream');
    
    socket.emit('stop_stream', {}, (response) => {
        if (response.success) {
            console.log('Stream stopped successfully');
            updateStreamStatus(false);
            stopConnectionTimer();
        } else {
            console.error('Error stopping stream');
            alert('Error stopping stream');
        }
    });
}

// Update connection status
function updateConnectionStatus(status) {
    connectionStatus.className = `status-indicator ${status}`;
    
    const statusText = connectionStatus.querySelector('.status-text');
    statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}

// Update stream status
function updateStreamStatus(active) {
    isStreaming = active;
    
    if (active) {
        videoPlaceholder.style.display = 'none';
        videoStream.style.display = 'block';
        startStreamBtn.disabled = true;
        stopStreamBtn.disabled = false;
        updateConnectionStatus('connected');
    } else {
        videoPlaceholder.style.display = 'block';
        videoStream.style.display = 'none';
        startStreamBtn.disabled = false;
        stopStreamBtn.disabled = true;
        updateConnectionStatus('disconnected');
    }
}

// Update configuration display
function updateConfigDisplay(config) {
    serverIpInput.value = config.server_ip || '';
    udpPortInput.value = config.udp_port || 8888;
    bufferSizeInput.value = config.buffer_size || 10;
    updateBufferSizeDisplay();
}

// Update statistics display
function updateStatsDisplay(stats) {
    fpsElement.textContent = stats.fps.toFixed(1);
    dataRateElement.textContent = stats.data_rate.toFixed(2) + ' Mbps';
    resolutionElement.textContent = stats.resolution;
    bitrateElement.textContent = stats.data_rate.toFixed(2) + ' Mbps';
    resolutionDisplay.textContent = stats.resolution;
}

// Start connection timer
function startConnectionTimer() {
    connectionStartTime = new Date();
    
    if (connectionTimer) {
        clearInterval(connectionTimer);
    }
    
    connectionTimer = setInterval(() => {
        const now = new Date();
        const elapsed = now - connectionStartTime;
        
        const hours = Math.floor(elapsed / 3600000).toString().padStart(2, '0');
        const minutes = Math.floor((elapsed % 3600000) / 60000).toString().padStart(2, '0');
        const seconds = Math.floor((elapsed % 60000) / 1000).toString().padStart(2, '0');
        
        connectionTimeElement.textContent = `${hours}:${minutes}:${seconds}`;
    }, 1000);
}

// Stop connection timer
function stopConnectionTimer() {
    if (connectionTimer) {
        clearInterval(connectionTimer);
        connectionTimer = null;
    }
    
    connectionTimeElement.textContent = '00:00:00';
}
