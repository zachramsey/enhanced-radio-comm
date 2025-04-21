# MJPG-Streamer Setup & Streaming Guide

This guide covers how to install and build MJPG‑streamer on Orange Pi running DietPi, stream USB camera feed over HTTP to a browser and outlines ideas for developing an application that toggles between streaming (acting as a server) and receiving (acting as a client) modes on both the Orange Pi and my laptop.

---

## Table of Contents

1. [Overview](#overview)
2. [Installing Dependencies](#installing-dependencies)
3. [Cloning & Building MJPG-Streamer](#cloning--building-mjpg-streamer)
4. [Running MJPG-Streamer and Accessing the Stream](#running-mjpg-streamer-and-accessing-the-stream)
5. [Configuring a mDNS on the Orange Pi](#configuring-a-mdns-on-the-orange-pi)
6. [Next Steps: Creating a Mode-Switching App](#next-steps-creating-a-mode-switching-app)
8. [References](#references)

---

## Overview

MJPG‑streamer captures video from your USB webcam, encodes each frame as JPEG, and streams these frames over HTTP.

---

## Installing Dependencies

I used `libjpeg62-turbo-dev`to run the and compile the MJPEG library. Run the following commands:

```bash
sudo apt-get update
sudo apt-get install -y cmake libjpeg62-turbo-dev imagemagick git build-essential gcc g++
```

---

## Cloning & Building MJPG Streamer

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/jacksonliam/mjpg-streamer.git
   ```

2. **Change to the Experimental Folder:**

   ```bash
   cd mjpg-streamer/mjpg-streamer-experimental
   ```

3. **Compile MJPG‑Streamer:**

   ```bash
   make clean all
   ```

4. **Install (Optional – for system-wide access if you have multiple users):**

   ```bash
   sudo make install
   ```

   > **Note:** If you run into library search path issues, set:
   >
   > ```bash
   > export LD_LIBRARY_PATH=$(pwd)
   > ```

---

## Running MJPG Streamer and Accessing the Stream

Assuming your camera appears as `/dev/video1`. You can launch MJPG‑streamer with the following command:

```bash
./mjpg_streamer -i "input_uvc.so -d /dev/video1 -r 640x480 -f 30" -o "output_http.so -w ./www"
```

- **Check which file the camera points too:**
  ```bash
  lsusb
  lsmod | grep uvcvideo
  ```

- **Explanation:**
  - `-d /dev/video1` specifies the camera device.
  - `-r 640x480` sets the resolution.
  - `-f 30` sets the framerate (frames per second).
  - The HTTP output serves content from the `./www` folder (default port is 8080).

### Testing in a Browser

1. **Open the Stream:**  
   On your laptop’s browser, navigate to:
   ```
   http://DietPi.local:8080/?action=stream
   ```

---

## Configuring a mDNS on the Orange Pi

To address the random IP address that gets changed every time you reconnect. I mad the PI accessible with just it's hostname.

Here's the steps I took to do that:

1. **Install Avahi (mDNS Daemon):**
   ```bash
   apt update
   ```

   ```bash
   apt install avahi-daemon
   ```

2. **Enable and start Avahi:**
   ```bash
   systemctl enable avahi-daemon
   ```

   ```bash
   systemctl start avahi-daemon
   ```

3. **Verify Avahi is running:**
   ```bash
   systemctl status avahi-daemon
   ```

## Next Steps: Creating a Mode Switching App

Imagine an app that can toggle the roles of the devices between “streamer” and “viewer”. The stepps and actions to be taken is still being studied. will update this section once I get a basic understanding of how this will work.

---

## References

- [MJPG-streamer GitHub Repository](https://github.com/jacksonliam/mjpg-streamer)
- [DietPi Network Configuration](https://dietpi.com/)

---
