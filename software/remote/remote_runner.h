
/*

*/

#ifndef REMOTE_RUNNER_H
#define REMOTE_RUNNER_H

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using namespace executorch::aten;
using namespace ::executorch::extension;

class RemoteRunner {
public:
    Module videoEncoder;    // Module for video processing
    Module controlDecoder;  // Module for control processing
    int rawWidth;           // Width of the input image
    int rawHeight;          // Height of the input image
    int latHeight;          // Height of the latent space
    int latWidth;           // Width of the latent space

    // Initialize the RemoteRunner with the paths to the video and control models
    RemoteRunner(
        const std::string& videoModelPath,      // Path to the exported video model
        const std::string& controlModelPath,    // Path to the exported control model
        const int rawWidth,                     // Width of the input image
        const int rawHeight                     // Height of the input image
    );

    // Compress an image using the video model
    uint8_t* encodeImage(const uint8_t* rawImage);

    // // Decompress control data using the control model
    // uint8_t* decodeControl(const uint8_t* latControl);
}

#endif // REMOTE_RUNNER_H