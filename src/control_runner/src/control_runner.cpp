
/*

*/

#include "control_runner.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <vector>
#include <memory>

using namespace ::executorch::aten;
using namespace ::executorch::extension;


/* --- Private implementation of ControlRunner --- */

struct ControlRunnerImpl {
    Module videoDecoder;    // Module for video processing

    int latHypH;            // Height of the latent hyper
    int latHypW;            // Width of the latent hyper
    int latHypC;            // Number of channels in the latent hyper
    int latHypSize;         // Size of the latent hyper

    int latImgH;            // Height of the latent image
    int latImgW;            // Width of the latent image
    int latImgC;            // Number of channels in the latent image
    int latImgSize;         // Size of the latent image

    // Initialize the RemoteRunner
    ControlRunnerImpl(
        const std::string& videoModelPath,  // Path to the exported video model
        int latHyperHeight,                 // Height of the latent hyper
        int latHyperWidth,                  // Width of the latent hyper
        int latHyperChannel,                // Number of channels in the latent hyper
        int latImageHeight,                 // Height of the latent image
        int latImageWidth,                  // Width of the latent image
        int latImageChannel                 // Number of channels in the latent image
    ) : videoDecoder(videoModelPath),
        latHypH(latHyperHeight),
        latHypW(latHyperWidth),
        latHypC(latHyperChannel),
        latImgH(latImageHeight),
        latImgW(latImageWidth),
        latImgC(latImageChannel)
    {
        // Load the video model
        const auto model_error = this->videoDecoder.load();

        // Load the decode method from the video model
        const auto method_error = this->videoDecoder.load_forward();

        // Calculate the number of bytes in the latent hyper and latent image
        this->latHypSize = latHyperHeight * latHyperWidth * latHyperChannel;
        this->latImgSize = latImageHeight * latImageWidth * latImageChannel;
    }

    // Decompress latent data using the video decoder model
    std::vector<uint8_t> decodeImageImpl(const std::vector<uint8_t>& data) {
        
        // Split the input data into latent hyper and latent image parts
        std::vector<uint8_t> latHypVector(data.begin(), data.begin() + this->latHypSize);
        std::vector<uint8_t> latImgVector(data.begin() + this->latHypSize, data.begin() + this->latHypSize + this->latImgSize);

        // Create tensor pointers for the latent hyper and latent image
        auto latHypTensor = make_tensor_ptr({this->latHypH, this->latHypW, this->latHypC}, latHypVector, ScalarType::Byte);
        auto latImgTensor = make_tensor_ptr({this->latImgH, this->latImgW, this->latImgC}, latImgVector, ScalarType::Byte);

        // Run the video model
        const auto result = this->videoDecoder.execute("forward", {latHypTensor, latImgTensor});

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

// Constructor for ControlRunner
ControlRunner::ControlRunner(
    const std::string& videoModelPath,
    int latImageHeight,
    int latImageWidth,
    int latImageChannel,
    int latHyperHeight,
    int latHyperWidth,
    int latHyperChannel
) : controlRunnerImpl(std::make_unique<ControlRunnerImpl>(
        videoModelPath,
        latHyperHeight,
        latHyperWidth,
        latHyperChannel,
        latImageHeight,
        latImageWidth,
        latImageChannel
    )) {}

// Destructor for ControlRunner
ControlRunner::~ControlRunner() = default;

// Move constructor for ControlRunner
ControlRunner::ControlRunner(ControlRunner&&) noexcept = default;

// Move assignment operator for ControlRunner
ControlRunner& ControlRunner::operator=(ControlRunner&&) noexcept = default;

// Decompress latent data using the video model
std::vector<uint8_t> ControlRunner::decodeImage(const std::vector<uint8_t>& data) {
    return this->controlRunnerImpl->decodeImageImpl(data);
}