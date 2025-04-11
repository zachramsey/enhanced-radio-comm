
/*

*/

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using namespace executorch::aten;
using namespace ::executorch::extension;

class ControlRunner {
public:
    Module videoDecoder;    // Module for video processing
    Module controlEncoder;  // Module for control processing
    int latHeight;          // Height of the latent image
    int latWidth;           // Width of the latent image
    int latChannels;        // Number of channels in the latent image

    // Initialize the RemoteRunner
    ControlRunner(
        const std::string& videoModelPath,      // Path to the exported video model
        const std::string& controlModelPath,    // Path to the exported control model
        const int latHeight,                    // Height of the latent image
        const int latWidth,                     // Width of the latent image
        const int latChannels                   // Number of channels in the latent image
    ) {
        // Instantiate the video and control models
        self->videoDecoder = Module(videoModelPath);
        self->controlEncoder = Module(controlModelPath);

        // Load the video model
        const auto error = videoDecoder.load();
        assert(videoDecoder.is_loaded());

        // Load the decode method from the video model
        const auto error = videoDecoder.load_method("decode");
        assert(module.is_method_loaded("decode"));

        // Load the control model
        const auto error = controlEncoder.load();
        assert(controlEncoder.is_loaded());

        // Load the encode method from the control model
        const auto error = controlEncoder.load_method("encode");
        assert(controlEncoder.is_method_loaded("encode"));

        // Store the width, height, and channels
        this->latWidth = latWidth;
        this->latHeight = latHeight;
        this->latChannels = latChannels;
    }

    // Decompress latent image data using the video model
    uint8_t* decodeImage(const uint8_t* latImage) {
        // Create a tensor from the input image
        auto input = from_blob(latImage, {self->latHeight, self->latWidth, self.latChannels}, ScalarType::Byte);
        // Run the video model
        const auto output = videoDecoder.get("decode", {input});
        return output;
    }

    // // Compress raw control data using the control model
    // uint8_t* encodeControl(const uint8_t* rawControl);
}
