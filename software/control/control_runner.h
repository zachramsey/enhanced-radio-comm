
/*

*/

#ifndef CONTROL_RUNNER_H
#define CONTROL_RUNNER_H

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

    // Initialize the RemoteRunner with the paths to the video and control models
    ControlRunner(
        const std::string& videoModelPath,      // Path to the exported video model
        const std::string& controlModelPath,    // Path to the exported control model
        const int latWidth,                     // Width of the latent image
        const int latHeight                     // Height of the latent image
    );

    // Decompress latent image data using the video model
    uint8_t* decodeImage(const uint8_t* latImage);

    // // Compress raw control data using the control model
    // uint8_t* encodeControl(const uint8_t* inputControl);
}

#endif // CONTROL_RUNNER_H