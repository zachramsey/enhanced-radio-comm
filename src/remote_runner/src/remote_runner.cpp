
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
    Module encoder;                 // Module for video processing

    std::vector<int> imgShape;      // Shape of the input image
    std::vector<int> latHypShape;   // Shape of the latent hyper
    std::vector<int> latImgShape;   // Shape of the latent image

    int imgSize;                    // Size of the input image
    int latHypSize;                 // Size of the latent hyper
    int latImgSize;                 // Size of the latent image

    // Initialize the RemoteRunner
    RemoteRunnerImpl(
        const std::string& modelPath,   // Path to the exported video model
        const int imgH,                       // Height of the input image
        const int imgW,                       // Width of the input image
        const int imgC,                       // Number of channels in the input image
        const int latHypH,                    // Height of the latent hyper
        const int latHypW,                    // Width of the latent hyper
        const int latHypC,                    // Number of channels in the latent hyper
        const int latImgH,                    // Height of the latent image
        const int latImgW,                    // Width of the latent image
        const int latImgC                     // Number of channels in the latent image
    ) : encoder(modelPath),
        imgShape({imgH, imgW, imgC}),
        latHypShape({latHypH, latHypW, latHypC}),
        latImgShape({latImgH, latImgW, latImgC}),
        imgSize(imgH * imgW * imgC),
        latHypSize(latHypH * latHypW * latHypC),
        latImgSize(latImgH * latImgW * latImgC) {

        // Load the video model
        const auto model_error = this->encoder.load();

        // Load the decode method from the video model
        const auto method_error = this->encoder.load_forward();
    }

    // Compress image data using the video encoder model
    std::vector<int8_t> encodeImageImpl(const std::vector<uint8_t>& data) {
        
        // Set the model input
        auto inputTensor = make_tensor_ptr(this->imgShape, data, ScalarType::Byte);
        this->encoder.set_input("forward", inputTensor, 0);

        // Set the model outputs
        auto latHypTensor = make_tensor_ptr(this->latHypShape, ScalarType::Byte);
        this->encoder.set_output("forward", latHypTensor, 0);

        auto latImgTensor = make_tensor_ptr(this->latImgShape, ScalarType::Char);
        this->encoder.set_output("forward", latImgTensor, 1);

        // Run the video model
        const auto result = this->encoder.execute("forward");

        // Check if the model ran successfully
        if (result.ok()) {
            // Get location of the latent hyper and latent image tensors
            const int8_t* latHypPtr = latHypTensor->const_data_ptr<int8_t>();
            const int8_t* latImgPtr = latImgTensor->const_data_ptr<int8_t>();

            // Create output vector with the size of latent hyper and latent image
            std::vector<int8_t> outVector;
            outVector.reserve(this->latHypSize + this->latImgSize);

            // Insert the latent hyper and latent image data into the output vector
            outVector.insert(outVector.end(), latHypPtr, latHypPtr + this->latHypSize);
            outVector.insert(outVector.end(), latImgPtr, latImgPtr + this->latImgSize);

            return outVector;
        } else {
            return {};
        }
    }
};


/* --- Public interface for ControlRunner --- */

// Constructor for RemoteRunner
RemoteRunner::RemoteRunner(
    const std::string& modelPath,
    const int imgH,
    const int imgW,
    const int imgC,
    const int latHypH,
    const int latHypW,
    const int latHypC,
    const int latImgH,
    const int latImgW,
    const int latImgC
) : remoteRunnerImpl(std::make_unique<RemoteRunnerImpl>(
        modelPath, imgH, imgW, imgC, latHypH, latHypW, latHypC, latImgH, latImgW, latImgC)) {
}

// Destructor for RemoteRunner
RemoteRunner::~RemoteRunner() = default;

// Move constructor for RemoteRunner
RemoteRunner::RemoteRunner(RemoteRunner&&) noexcept = default;

// Move assignment operator for RemoteRunner
RemoteRunner& RemoteRunner::operator=(RemoteRunner&&) noexcept = default;

// Compress image data using the video encoder model
std::vector<int8_t> RemoteRunner::encodeImage(const std::vector<uint8_t>& data) const {
    return remoteRunnerImpl->encodeImageImpl(data);
}