
### Set up training environment
``` bash
pyenv virtualenv 3.12.8 compressai
pyenv local compressai

# Install CompressAI
pip3 install git+https://github.com/InterDigitalInc/CompressAI.git@torch_cpp_extension

# Install TorchEval
pip install --pre torcheval-nightly
pip3 install scikit-image   # Dependency of TorchEval

# Nvidia GPU
pip3 uninstall torch torchvision -y
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 -y

# AMD GPU
pip3 uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2.4
```

### Set up compiling environment
``` bash
pyenv virtualenv 3.12.8 executorch
pyenv shell executorch
cd software/
git clone --branch release/0.6 https://github.com/pytorch/executorch.git
cd executorch
git submodule sync && git submodule update --init
./install_executorch.sh

# Configuring CMake
mkdir cmake-out
cmake \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPTIMIZE_SIZE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DPYTHON_EXECUTABLE=python \
    -Bcmake-out .
cmake --build cmake-out -j9 --target install --config Release
```