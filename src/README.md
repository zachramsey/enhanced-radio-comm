
# 

---

## File Tree
```
src
├── CMakeLists.txt
├── ControlRunner
│   ├── cmake
│   │   └── ControlRunnerConfig.cmake.in
│   ├── CMakeLists.txt
│   ├── include
│   │   └── control_runner.h
│   └── src
│       └── control_runner.cpp
├── README.md
├── RemoteRunner
│   ├── cmake
│   │   └── RemoteRunnerConfig.cmake.in
│   ├── CMakeLists.txt
│   ├── include
│   │   └── remote_runner.h
│   └── src
│       └── remote_runner.cpp
├── setup.md
└── training
    ├── __init__.py
    ├── main.py
    └── video
        ├── bottlenecks.py
        ├── compiler.py
        ├── config.py
        ├── decoder copy.py
        ├── decoder.py
        ├── encoder copy.py
        ├── encoder.py
        ├── __init__.py
        ├── loader.py
        ├── model.py
        ├── simulate.py
        ├── trainer.py
        └── utils.py
```

---

## Setup

### Managing Python Environments *(pyenv + pyenv-virtualenv recommended)*

**Set up pyenv:**
``` bash
# Automatic Installer
curl -fsSL https://pyenv.run | bash

# Add Pyenv to your shell

# Add to ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc

# Also add to ~/.profile, ~/.bash_profile, and ~/.bash_login (if they exist).
# If none exist, create ~/.profile and add there.

# Restart your shell for changes to take effect
exec "$SHELL"

# Install Python 3.12.8
pyenv install 3.12.8
```

**Set up pyenv-virtualenv:**
``` bash
# Check out pyenv-virtualenv into pyenv plugin directory
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

# Add pyenv virtualenv-init to your shell
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Restart your shell for changes to take effect
exec "$SHELL"
```

### Setting up C++ libraries
**Prepare virtual environment:**
``` bash
# Create a virtual environment for executorch dependencies
pyenv virtualenv 3.12.8 executorch_build

# Activate environment (may require IDE-specific configuration)
pyenv activate executorch_build

# Ensure pip is up to date in the virtual environment
pip3 install --upgrade pip
```

**Build Executorch from source:**
``` bash
# Clone Executorch
cd src
git clone --branch release/0.6 https://github.com/pytorch/executorch.git

# Update Executorch submodules
cd executorch
git submodule sync && git submodule update --init

# Install Executorch
./install_executorch.sh
cd ..   # Back to src
```

**Build C++ libraries with Executorch runtime:**
``` bash
cmake -B build .
cmake --build build -j"$(nproc+1)"
sudo cmake --build build -j"$(nproc+1)" --target install
cd ..   # Back to root
```

### Setting up PyTorch training environment
**Prepare virtual environment:**
``` bash
# Create a virtual environment for executorch dependencies
pyenv virtualenv 3.12.8 enhanced_comm

# Activate environment (may require IDE-specific configuration)
pyenv activate enhanced_comm

# Ensure pip is up to date in the virtual environment
pip3 install --upgrade pip
```

**Install training requirements**
``` bash
pip3 install --pre torcheval-nightly
pip3 install scikit-image compressai executorch

# Install PyTorch from https://pytorch.org/get-started/locally/
```

---