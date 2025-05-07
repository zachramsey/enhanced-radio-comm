# Executorch APIs (C++ Libraries)

## Managing Python Environments *(pyenv + pyenv-virtualenv method)*

### Set up pyenv
``` bash
# Automatic Pyenv Installer
curl -fsSL https://pyenv.run | bash

# --- Add Pyenv to your shell ---
# Add to ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
# Also add to ~/.profile, ~/.bash_profile, and ~/.bash_login (if they exist).
# If none exist, create ~/.profile and add there.

# Restart your shell for changes to take effect
exec "$SHELL"
```

### Set up pyenv-virtualenv
``` bash
# Check out pyenv-virtualenv into pyenv plugin directory
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

# Add pyenv virtualenv-init to your shell
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Restart your shell for changes to take effect
exec "$SHELL"
```

## Compiling The APIs
*Note: The APIs are pre-built in "libmonkeycomm.tar.gz." This file can be decompressed in-place and the APIs used out-of-the-box.*

### Prepare virtual environment
``` bash
# Install Python
pyenv install 3.12.8

# Create a virtual environment for executorch dependencies
pyenv virtualenv 3.12.8 executorch_build

# Activate environment (may require IDE-specific configuration)
pyenv activate executorch_build

# Ensure pip is up to date in the virtual environment
pip3 install --upgrade pip
```

### Prepare Executorch source code
``` bash
# Clone Executorch
cd src
git clone --branch release/0.6 https://github.com/pytorch/executorch.git

# Update Executorch submodules
cd executorch
git submodule sync
git submodule update --init

# Install Executorch requirements
./install_requirements.sh
cd ../../
```

### Build C++ libraries with Executorch runtime
``` bash
cmake -B build .
cmake --build build -j"$(nproc+1)" --target install
```

## Usage

### Compiling with the APIs
Once built, the APIs for both control and remote deice are accessible. The following prototype shell script outlines how an application may be compiled with the libraries.
``` bash
# Assuming current directory contains your source file
MONKEY_COMMS_DIR="/path/to/libmonkeycomms"
INCLUDE_DIR="${MONKEY_COMMS_DIR}/include"
LIB_DIR="${MONKEY_COMMS_DIR}/lib"

# Compile application
g++ -o my_application main.cpp \
    -I${INCLUDE_DIR} \
    -L${LIB_DIR} \
    -lcontrol_runner \
    -lremote_runner \
    -std=c++17 \
    -Wl,-rpath,${LIB_DIR}

# Run the application
./my_application
```

### Initialization
``` cpp
#include <control_runner.h>
#include <remote_runner.h>

// Model executable paths
const std::string imgEncPath = "path/to/img_enc_model.pte";
const std::string imgEncPath = "path/to/img_enc_model.pte";

// Common initialization parameters
const int imgHeight = 480;      // Input/output image height
const int imgWidth = 640;       // Input/output image width
const int imgChannels = 3;      // RGB channels
const int latHypHeight = 30;    // Latent hypothesis height
const int latHypWidth = 40;     // Latent hypothesis width
const int latHypChannels = 192; // Latent hypothesis channels
const int latImgHeight = 60;    // Latent image height
const int latImgWidth = 80;     // Latent image width
const int latImgChannels = 192; // Latent image channels

// Initialize encoder
RemoteRunner encoder(
    imgEncPath,
    imgHeight, imgWidth, imgChannels,
    latHypHeight, latHypWidth, latHypChannels,
    latImgHeight, latImgWidth, latImgChannels
);

// Initialize decoder
ControlRunner decoder(
    imgDecPath,
    imgHeight, imgWidth, imgChannels,
    latHypHeight, latHypWidth, latHypChannels,
    latImgHeight, latImgWidth, latImgChannels
);
```

### Encoding a video frame
``` cpp
// Input: Vector of uint8_t (bytes) containing raw RGB888 pixel data
// Output: Compressed frame data as vector of int8_t (chars)
std::vector<uint8_t> frameData = captureFrame(); // Your frame data source
std::vector<int8_t> compressedData = remoteRunner.encodeImage(frameData);
```

### Decoding a video frame
``` cpp
// Input: Vector of int8_t (chars) containing compressed data from RemoteRunner
// Output: Reconstructed RGB888 frame as vector of uint8_t (bytes)
std::vector<int8_t> compressedData = receiveData(); // From network, etc.
std::vector<uint8_t> decodedImage = controlRunner.decodeImage(compressedData);
```
