#!/usr/bin/env bash
set -e  # Exit on error
set -u  # Treat unset vars as error

PYTHON_VERSION="3.12.8"
EXECUTORCH_ENV="executorch_build"
MAIN_ENV="enhanced_comm"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Append line to shell config files if missing
append_to_shell_configs() {
    local line="$1"
    for file in ~/.bashrc ~/.profile ~/.bash_profile ~/.bash_login; do
        if [ -f "$file" ]; then
            grep -qxF "$line" "$file" || echo "$line" >> "$file"
        fi
    done
}

echo ">> Checking for pyenv and pyenv-virtualenv..."

# Install pyenv if missing
if ! hash pyenv >/dev/null 2>&1; then
    echo ">> Installing pyenv..."

    curl -fsSL https://pyenv.run | bash

    # Add pyenv to shell config files
    append_to_shell_configs 'export PYENV_ROOT="$HOME/.pyenv"'
    append_to_shell_configs '[[ -d "$PYENV_ROOT/bin" ]] && export PATH="$PYENV_ROOT/bin:$PATH"'
    append_to_shell_configs 'eval "$(pyenv init --path)"'
    append_to_shell_configs 'eval "$(pyenv init - bash)"'
    append_to_shell_configs 'eval "$(pyenv virtualenv-init -)"'

    # Apply changes for current shell session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init - bash)"
    eval "$(pyenv virtualenv-init -)"
else
    # Ensure runtime environment still gets pyenv loaded even if already installed
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init - bash)"
    eval "$(pyenv virtualenv-init -)"
fi

# Install pyenv-virtualenv if missing
if [ ! -d "$(pyenv root)/plugins/pyenv-virtualenv" ]; then
    echo ">> Installing pyenv-virtualenv..."
    git clone https://github.com/pyenv/pyenv-virtualenv.git "$(pyenv root)/plugins/pyenv-virtualenv"
    eval "$(pyenv virtualenv-init -)"
fi

# Ensure Python version is available
if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
    echo ">> Installing Python $PYTHON_VERSION..."
    pyenv install "$PYTHON_VERSION"
fi

# --------------------------
# STEP 1: Executorch setup
# --------------------------

echo ">> Setting up Executorch build environment..."
if ! pyenv versions | grep -q "$EXECUTORCH_ENV"; then
    pyenv virtualenv "$PYTHON_VERSION" "$EXECUTORCH_ENV"
fi
pyenv activate "$EXECUTORCH_ENV"
pip3 install --upgrade pip

cd src
if [ ! -d "executorch" ]; then
    echo ">> Cloning Executorch..."
    git clone --branch release/0.6 https://github.com/pytorch/executorch.git
fi

echo ">> Updating Executorch submodules..."
cd executorch
git submodule sync && git submodule update --init

echo ">> Installing Executorch..."
./install_executorch.sh

echo ">> Building Executorch..."
mkdir -p cmake-out
cmake \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPTIMIZE_SIZE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DPYTHON_EXECUTABLE=$(which python) \
    -Bcmake-out .

cmake --build cmake-out -j"$(nproc+1)" --target install --config Release

cd ../../  # back to root
pyenv deactivate

# --------------------------
# STEP 2: Main environment
# --------------------------

echo ">> Setting up main environment..."
if ! pyenv versions | grep -q "$MAIN_ENV"; then
    pyenv virtualenv "$PYTHON_VERSION" "$MAIN_ENV"
fi
pyenv activate "$MAIN_ENV"
pip3 install --upgrade pip

# Install rest of dependencies
echo ">> Installing remaining packages..."
pip3 install --pre torcheval-nightly
pip3 install scikit-image
pip3 install git+https://github.com/InterDigitalInc/CompressAI.git@torch_cpp_extension

# Detect CUDA or ROCm
pip3 uninstall -y torch torchvision
if hash nvidia-smi >/dev/null 2>&1; then
    echo ">> Nvidia GPU detected; installing CUDA-enabled PyTorch..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
elif hash rocminfo >/dev/null 2>&1; then
    echo ">> AMD GPU detected; installing ROCm-enabled PyTorch..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2.4
else
    echo ">> No compatible GPU detected; installing CPU-only PyTorch..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Copy built executorch from executorch_env
echo ">> Copying Executorch build into main environment..."
EXECUTORCH_SITE=$(pyenv prefix "$EXECUTORCH_ENV")/lib/python${PYTHON_VERSION%.*}/site-packages
cp -r "$EXECUTORCH_SITE"/executorch* "$(pyenv prefix)/lib/python${PYTHON_VERSION%.*}/site-packages/"

# Install missing executorch dependencies
echo ">> Installing Executorch dependencies..."
pip3 install expecttest flatbuffers hypothesis pandas parameterized pytest pytest-rerunfailures pytest-xdist pyyaml ruamel-yaml tabulate torchao

# Clean up
echo ">> Cleaning up..."
cd src/executorch
git clean -fdx
cd ../../
rm -rf src/executorch
echo ">> Cleaning up pyenv..."
pyenv uninstall -f executorch_build

echo "Environment setup complete. Active environment: $MAIN_ENV"
