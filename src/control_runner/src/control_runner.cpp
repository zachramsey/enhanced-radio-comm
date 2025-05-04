
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
    Module videoDecoder;            // Module for video processing

    std::vector<int> imgShape;      // Shape of the latent hyper
    std::vector<int> latHypShape;   // Shape of the latent hyper
    std::vector<int> latImgShape;   // Shape of the latent image

    int imgSize;                    // Size of the input image
    int latHypSize;                 // Size of the latent hyper
    int latImgSize;                 // Size of the latent image

    // Initialize the RemoteRunner
    ControlRunnerImpl(
        const std::string& videoModelPath,  // Path to the exported video model
        const int imgH,                           // Height of the input image
        const int imgW,                           // Width of the input image
        const int imgC,                           // Number of channels in the input image
        const int latHypH,                        // Height of the latent hyper
        const int latHypW,                        // Width of the latent hyper
        const int latHypC,                        // Number of channels in the latent hyper
        const int latImgH,                        // Height of the latent image
        const int latImgW,                        // Width of the latent image
        const int latImgC                         // Number of channels in the latent image
    ) : videoDecoder(videoModelPath) {
        // Load the video model
        const auto model_error = this->videoDecoder.load();

        // Load the decode method from the video model
        const auto method_error = this->videoDecoder.load_forward();

        // Set the shapes for the input image, output latent hyper, and output latent image
        this->imgShape = {imgH, imgW, imgC};
        this->latHypShape = {latHypH, latHypW, latHypC};
        this->latImgShape = {latImgH, latImgW, latImgC};

        // Calculate the number of bytes in the input image, latent hyper, and latent image
        this->imgSize = imgH * imgW * imgC;
        this->latHypSize = latHypH * latHypW * latHypC;
        this->latImgSize = latImgH * latImgW * latImgC;
    }

    // Decompress latent data using the video decoder model
    std::vector<uint8_t> decodeImageImpl(const std::vector<uint8_t>& data) {
        
        // Split the input data into latent hyper and latent image parts
        std::vector<uint8_t> latHypVector(data.begin(), data.begin() + this->latHypSize);
        std::vector<uint8_t> latImgVector(data.begin() + this->latHypSize, data.begin() + this->latHypSize + this->latImgSize);

        // Set the model inputs
        auto latHypTensor = make_tensor_ptr(this->latHypShape, latHypVector, ScalarType::Byte);
        this->videoDecoder.set_input("forward", latHypTensor, 0);

        auto latImgTensor = make_tensor_ptr(this->latImgShape, latImgVector, ScalarType::Byte);
        this->videoDecoder.set_input("forward", latImgTensor, 1);

        // Set the model output
        auto imgTensor = make_tensor_ptr(this->imgShape, ScalarType::Byte);
        this->videoDecoder.set_output("forward", imgTensor, 0);

        // Run the video model
        const auto result = this->videoDecoder.execute("forward");

        // Check if the model ran successfully
        if (result.ok()) {
            // Get the output tensor
            const uint8_t* imgPtr = imgTensor->const_data_ptr<uint8_t>();
            std::vector<uint8_t> outVector(imgPtr, imgPtr + this->imgSize);
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
    const int imgH,
    const int imgW,
    const int imgC,
    const int latHypH,
    const int latHypW,
    const int latHypC,
    const int latImgH,
    const int latImgW,
    const int latImgC
) : controlRunnerImpl(std::make_unique<ControlRunnerImpl>(
        videoModelPath,
        imgH,
        imgW,
        imgC,
        latHypH,
        latHypW,
        latHypC,
        latImgH,
        latImgW,
        latImgC
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