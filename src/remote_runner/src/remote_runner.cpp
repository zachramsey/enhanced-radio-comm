
/*

*/

#include "remote_runner.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <vector>
#include <memory>

using namespace ::executorch::aten;
using namespace ::executorch::extension;


/* --- Private implementation of RemoteRunner --- */

struct RemoteRunnerImpl {
    Module videoEncoder;    // Module for video processing
    int imgH;               // Height of the latent hyper
    int imgW;               // Width of the latent hyper
    int imgC;               // Number of channels in the latent hyper
    int imgSize;            // Size of the latent hyper

    // Initialize the RemoteRunner
    RemoteRunnerImpl(
        const std::string& videoModelPath,  // Path to the exported video model
        int imageHeight,              // Height of the input image
        int imageWidth,               // Width of the input image
        int imageChannel              // Number of channels in the input image
    ) : videoEncoder(videoModelPath),
        imgH(imageHeight),
        imgW(imageWidth),
        imgC(imageChannel)
    {
        // Load the video model
        const auto model_error = this->videoEncoder.load();

        // Load the decode method from the video model
        const auto method_error = this->videoEncoder.load_forward();

        // Calculate the number of bytes in the iinput image
        this->imgSize = imageHeight * imageWidth * imageChannel;
    }

    // Compress image data using the video encoder model
    std::vector<uint8_t> encodeImageImpl(const std::vector<uint8_t>& data) {
        
        // Create tensor pointers for the input image
        auto imgTensor = make_tensor_ptr({this->imgH, this->imgW, this->imgC}, data, ScalarType::Byte);

        // Run the video model
        const auto result = this->videoEncoder.execute("forward", imgTensor);

        // Check if the model ran successfully
        if (result.ok()) {
            // Get the output tensor and convert it to a vector
            const auto& output = result.get().at(0).toTensor();
            const uint8_t* outPtr = output.const_data_ptr<uint8_t>();
            std::vector<uint8_t> outVector(outPtr, outPtr + output.numel());
            return outVector;
        } else {
            return {};
        }
    }
};


/* --- Public interface for ControlRunner --- */

// Constructor for RemoteRunner
RemoteRunner::RemoteRunner(
    const std::string& videoModelPath,
    int imageHeight,
    int imageWidth,
    int imageChannel
) : remoteRunnerImpl(std::make_unique<RemoteRunnerImpl>(
        videoModelPath, 
        imageHeight, 
        imageWidth, 
        imageChannel
    )) {}

// Destructor for RemoteRunner
RemoteRunner::~RemoteRunner() = default;

// Move constructor for RemoteRunner
RemoteRunner::RemoteRunner(RemoteRunner&&) noexcept = default;

// Move assignment operator for RemoteRunner
RemoteRunner& RemoteRunner::operator=(RemoteRunner&&) noexcept = default;

// Compress image data using the video encoder model
std::vector<uint8_t> RemoteRunner::encodeImage(const std::vector<uint8_t>& data) const {
    return remoteRunnerImpl->encodeImageImpl(data);
}