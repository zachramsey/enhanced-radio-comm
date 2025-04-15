/**
 * Enhanced Radio Communication - Control Device JavaScript
 *
 * This script handles:
 * - Device discovery and connection
 * - Video stream display
 * - Data usage statistics
 * - User interface interactions
 */

// DOM Elements
const connectionStatus = document.getElementById('connection-status');
const deviceList = document.getElementById('device-list');
const refreshDevicesBtn = document.getElementById('refresh-devices');
const startStreamBtn = document.getElementById('start-stream');
const stopStreamBtn = document.getElementById('stop-stream');
const qualitySlider = document.getElementById('quality-slider');
const qualityValue = document.getElementById('quality-value');
const videoPlaceholder = document.getElementById('video-placeholder');
const videoStream = document.getElementById('video-stream');
const currentBitrate = document.getElementById('current-bitrate');
const totalDataReceived = document.getElementById('total-data-received');
const connectionTime = document.getElementById('connection-time');
const setupModal = document.getElementById('setup-modal');
const autoLaunchCheckbox = document.getElementById('auto-launch');
const saveSettingsBtn = document.getElementById('save-settings');

// Global variables
let selectedDevice = null;
let isStreaming = false;
let connectionStartTime = null;
let statsInterval = null;
let streamElement = null;
let totalBytes = 0;

// Check if this is the first run
const isFirstRun = localStorage.getItem('firstRun') !== 'false';
if (isFirstRun) {
    setupModal.classList.add('show');
}

// Close modal when ESC key is pressed
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && setupModal.classList.contains('show')) {
        closeSetupModal();
    }
});

// Function to close the setup modal
function closeSetupModal() {
    setupModal.classList.remove('show');
    localStorage.setItem('firstRun', 'false');
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the application
    fetchDevices();

    // Set up event listeners
    refreshDevicesBtn.addEventListener('click', fetchDevices);
    startStreamBtn.addEventListener('click', startStream);
    stopStreamBtn.addEventListener('click', stopStream);
    qualitySlider.addEventListener('input', updateQualityValue);
    saveSettingsBtn.addEventListener('click', saveSettings);

    // Add event listener for the close button
    const closeButton = document.querySelector('.close-button');
    if (closeButton) {
        closeButton.addEventListener('click', closeSetupModal);
    }

    // Start periodic stats updates
    statsInterval = setInterval(updateStats, 1000);
});

// Functions

/**
 * Fetch available remote devices
 */
function fetchDevices() {
    deviceList.innerHTML = '<p>Searching for devices...</p>';

    fetch('/api/devices')
        .then(response => response.json())
        .then(devices => {
            if (devices.length === 0) {
                deviceList.innerHTML = '<p>No devices found. Click refresh to try again.</p>';
                return;
            }

            deviceList.innerHTML = '';
            let orangePiFound = false;

            devices.forEach(device => {
                const deviceElement = document.createElement('div');
                deviceElement.className = 'device-item';
                deviceElement.dataset.address = device.address;
                deviceElement.dataset.port = device.port;
                deviceElement.innerHTML = `
                    <strong>${device.name || 'Unknown Device'}</strong><br>
                    <small>${device.address}:${device.port}</small>
                `;

                // Check if this is the Orange Pi
                if (device.address === 'dietpi.local' && device.port === 8080) {
                    deviceElement.classList.add('orange-pi');
                    orangePiFound = true;
                }

                deviceElement.addEventListener('click', () => selectDevice(deviceElement, device));
                deviceList.appendChild(deviceElement);

                // Auto-select the Orange Pi if found
                if (device.address === 'dietpi.local' && device.port === 8080) {
                    setTimeout(() => selectDevice(deviceElement, device), 500);
                }
            });

            // Add a note if Orange Pi is found
            if (orangePiFound) {
                const noteElement = document.createElement('div');
                noteElement.className = 'device-note';
                noteElement.innerHTML = '<small>✓ Orange Pi automatically connected</small>';
                deviceList.appendChild(noteElement);
            }
        })
        .catch(error => {
            console.error('Error fetching devices:', error);
            deviceList.innerHTML = '<p>Error fetching devices. Please try again.</p>';
        });
}

/**
 * Select a remote device
 */
function selectDevice(element, device) {
    // Remove selection from previously selected device
    const previouslySelected = document.querySelector('.device-item.selected');
    if (previouslySelected) {
        previouslySelected.classList.remove('selected');
    }

    // Mark this device as selected
    element.classList.add('selected');
    selectedDevice = device;

    // Enable the start stream button
    startStreamBtn.disabled = false;

    // Update connection status
    updateConnectionStatus('connecting');

    // Connect to the device
    connectToDevice(device);
}

/**
 * Connect to the selected remote device
 */
function connectToDevice(device) {
    fetch('/api/connect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            address: device.address,
            port: device.port
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateConnectionStatus('connected');
            connectionStartTime = new Date();
            startStreamBtn.disabled = false;
        } else {
            updateConnectionStatus('disconnected');
            console.error('Failed to connect:', data.error);
        }
    })
    .catch(error => {
        updateConnectionStatus('disconnected');
        console.error('Error connecting to device:', error);
    });
}

/**
 * Start the video stream
 */
function startStream() {
    if (!selectedDevice) {
        alert('Please select a device first');
        return;
    }

    isStreaming = true;
    startStreamBtn.disabled = true;
    stopStreamBtn.disabled = false;

    // Hide placeholder and show video container
    videoPlaceholder.style.display = 'none';
    videoStream.style.display = 'block';

    // Create and add the video element
    // For MJPEG streams, we use an img tag with a continuously updating src
    if (streamElement) {
        videoStream.removeChild(streamElement);
    }

    streamElement = document.createElement('img');

    // Use the correct URL format for the Orange Pi MJPG-Streamer
    if (selectedDevice.address === 'dietpi.local' && selectedDevice.port === 8080) {
        streamElement.src = `http://${selectedDevice.address}:${selectedDevice.port}/?action=stream&t=${new Date().getTime()}`;
        console.log(`Connecting to Orange Pi stream at ${streamElement.src}`);
    } else {
        // Generic format for other devices
        streamElement.src = `http://${selectedDevice.address}:${selectedDevice.port}/stream?t=${new Date().getTime()}`;
    }

    // Track data usage
    let frameCount = 0;
    const startTime = new Date().getTime();

    streamElement.onload = function() {
        frameCount++;

        // Calculate actual data rate based on frame count and time
        const currentTime = new Date().getTime();
        const elapsedSeconds = (currentTime - startTime) / 1000;

        if (elapsedSeconds >= 1) {
            const fps = frameCount / elapsedSeconds;
            // Rough estimate - in a real app, you'd want to use the actual transferred bytes
            const estimatedBytes = 30000; // Assume ~30KB per frame for 640x480 MJPEG
            updateDataUsage(estimatedBytes * fps);
        }
    };

    // Add error handling
    streamElement.onerror = function() {
        console.error('Error loading stream');
        videoPlaceholder.style.display = 'flex';
        videoPlaceholder.innerHTML = '<p>Error connecting to stream. Please try again.</p>';
    };

    videoStream.appendChild(streamElement);
}

/**
 * Stop the video stream
 */
function stopStream() {
    isStreaming = false;
    startStreamBtn.disabled = false;
    stopStreamBtn.disabled = true;

    // Remove the stream element
    if (streamElement) {
        videoStream.removeChild(streamElement);
        streamElement = null;
    }

    // Show placeholder and hide video container
    videoPlaceholder.style.display = 'flex';
    videoStream.style.display = 'none';
}

/**
 * Update the quality value display
 */
function updateQualityValue() {
    qualityValue.textContent = qualitySlider.value;

    // In a real implementation, you would send this value to the server
    // to adjust the stream quality
    if (isStreaming && selectedDevice) {
        // Example: fetch(`/api/quality?value=${qualitySlider.value}`);
        console.log(`Setting quality to ${qualitySlider.value}`);
    }
}

/**
 * Update connection status UI
 */
function updateConnectionStatus(status) {
    connectionStatus.className = `status-indicator ${status}`;

    const statusText = connectionStatus.querySelector('.status-text');
    statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}

/**
 * Update data usage statistics
 */
function updateDataUsage(bytes) {
    totalBytes += bytes;

    // Update the UI in the next stats update
}

/**
 * Update statistics display
 */
function updateStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(stats => {
            // Format and display bitrate
            const bitrate = stats.current_bitrate / 1000; // Convert to kbps
            currentBitrate.textContent = `${bitrate.toFixed(2)} kbps`;

            // Format and display total data
            const totalData = totalBytes / 1024; // Convert to KB
            if (totalData > 1024) {
                totalDataReceived.textContent = `${(totalData / 1024).toFixed(2)} MB`;
            } else {
                totalDataReceived.textContent = `${totalData.toFixed(2)} KB`;
            }

            // Update connection time if connected
            if (connectionStartTime) {
                const elapsed = new Date() - connectionStartTime;
                const hours = Math.floor(elapsed / 3600000).toString().padStart(2, '0');
                const minutes = Math.floor((elapsed % 3600000) / 60000).toString().padStart(2, '0');
                const seconds = Math.floor((elapsed % 60000) / 1000).toString().padStart(2, '0');
                connectionTime.textContent = `${hours}:${minutes}:${seconds}`;
            }
        })
        .catch(error => {
            console.error('Error fetching stats:', error);
        });
}

/**
 * Save user settings
 */
function saveSettings() {
    // Save auto-launch preference
    const autoLaunch = autoLaunchCheckbox.checked;

    fetch('/api/settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            auto_launch: autoLaunch
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Close the modal
            closeSetupModal();
        } else {
            console.error('Failed to save settings:', data.error);
            // Close the modal anyway to prevent it from being stuck
            closeSetupModal();
        }
    })
    .catch(error => {
        console.error('Error saving settings:', error);
        // Close the modal even if there's an error to prevent it from being stuck
        closeSetupModal();
    });
}
