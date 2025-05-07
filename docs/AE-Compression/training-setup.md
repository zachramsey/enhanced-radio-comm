# Model Trainer and Executable Compiler

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

## Set up training environment

### Prepare virtual environment
``` bash
# Create a virtual environment for executorch dependencies
pyenv virtualenv 3.12.8 enhanced_comm

# Activate environment (may require IDE-specific configuration)
pyenv activate enhanced_comm

# Ensure pip is up to date in the virtual environment
pip3 install --upgrade pip
```

### Install dependencies
``` bash
pip3 install --pre torcheval-nightly
pip3 install scikit-image compressai executorch

# Install PyTorch according to https://pytorch.org/get-started/locally/
```

## Usage

### Trainer
Set desired training configurations in config.py
Start the model trainer by running main.py with no command line arguments:
``` bash
python main.py
```

## 
Compile an existing model checkpoint to an Executorch executable (*.pte) with XNNPack backend by running main.py with command line arguments:
``` bash
python main.py <model_path> <c_network> <c_compress> <control_dir> <remote_dir> <quantize>
#  <model_path> | path to the model checkpoint file (path/to/model.pth)
#   <c_network> | number of network channel defined in config during model training
#  <c_compress> | number of compress channels defined in config during model training
# <control_dir> | path to directory to save control model executable
#  <remote_dir> | path to directory to save remote model executable
#    <quantize> | boolean indicating whether model should be quantized
```