
/*

*/

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using namespace executorch::aten;
using namespace ::executorch::extension;

class RemoteRunner {
public:
    Module videoEncoder;    // Module for video processing
    Module controlDecoder;  // Module for control processing
    int rawWidth;           // Width of the raw image input
    int rawHeight;          // Height of the raw image input

    // Initialize the RemoteRunner with the paths to the video and control models
    RemoteRunner(
        const std::string& videoModelPath,      // Path to the exported video model
        const std::string& controlModelPath,    // Path to the exported control model
        const int rawWidth,                        // Width of the input image
        const int rawHeight                        // Height of the input image
    ){
        // Instantiate the video and control models
        self->videoEncoder = Module(videoModelPath);
        self->controlDecoder = Module(controlModelPath);

        // Load the video model
        const auto error = videoEncoder.load();
        assert(videoEncoder.is_loaded());

        // Load the encode method from the video model
        const auto error = videoEncoder.load_method("encode");
        assert(module.is_method_loaded("encode"));

        // Load the control model
        const auto error2 = controlModule_.load();
        assert(controlModule_.is_loaded());

        // Load the decode method from the control model
        const auto error2 = controlModule_.load_method("decode");
        assert(controlModule_.is_method_loaded("decode"));

        // Store the width and height
        this->rawWidth = rawWidth;
        this->rawHeight = rawHeight;
    }

    // Compress raw image data using the video model
    uint8_t* encodeImage(const uint8_t* rawImage) {
        // Create a tensor from the input image
        auto input = from_blob(rawImage, {self->rawHeight, self->rawWidth, 3}, ScalarType::Byte);
        // Run the video model
        const auto output = videoEncoder.get("encode", {input});
        return output
    }
        
    // // Decompress latent control data using the control model
    // uint8_t* decodeControl(const uint8_t* rawData) {
    // }

