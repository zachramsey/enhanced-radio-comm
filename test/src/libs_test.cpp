
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <list>
#include <png.h>

#include <remote_runner.h>
#include <control_runner.h>
#include <lodepng.h>

const std::map<std::string, std::string> remotePTE = {
    {"small_q8", "../pte/remote/img_enc_small_xnnpack_q8.pte"},
    {"small_f32", "../pte/remote/img_enc_small_xnnpack_f32.pte"},
    {"default_q8", "../pte/remote/img_enc_default_xnnpack_q8.pte"},
    {"default_f32", "../pte/remote/img_enc_default_redux_xnnpack_f32.pte"}
};

const std::map<std::string, std::string> controlPTE = {
    {"small_q8", "../pte/control/img_enc_small_xnnpack_q8.pte"},
    {"small_f32", "../pte/control/img_enc_small_xnnpack_f32.pte"},
    {"default_q8", "../pte/control/img_enc_default_xnnpack_q8.pte"},
    {"default_f32", "../pte/control/img_enc_default_redux_xnnpack_f32.pte"}
};

int height = 480, width = 640;
const int latHypH = 8, latHypW = 10;
const int latImgH = 30, latImgW = 40;
const std::map<std::string, std::list<int>> latChannels = {
    {"small_q8", {64, 96}},
    {"small_f32", {64, 96}},
    {"default_q8", {128, 192}},
    {"default_f32", {128, 192}}
};

const std::string IMAGE_PATH = "res/test_image.png";
const std::string OUTPUT_PATH = "res/output_image.png";
const std::string PTE_TYPE = "default_f32";


#include <iostream>
#include <fstream>
#include <vector>
#include "lodepng.h"

// Structure to hold image data along with its dimensions
struct Rgb888Image {
    std::vector<uint8_t> data;
    unsigned width;
    unsigned height;
    std::string error_message;
};

Rgb888Image png_to_rgb888(const std::string& filename) {
    Rgb888Image result;
    uint8_t* buffer = nullptr;
    unsigned error;

    // Decode the PNG file directly to 24-bit RGB
    // LodePNG will allocate the buffer which must be freed later using free()
    error = lodepng_decode24_file(&buffer, &result.width, &result.height, filename.c_str());

    if (error) {
        result.error_message = "LodePNG decode error " + std::to_string(error) + ": " + lodepng_error_text(error);
        if (buffer) {
            free(buffer); // free buffer even if there was an error, if it was allocated
        }
        return result;
    }

    if (buffer) {
        // Calculate the size of the decoded image data (width * height * 3 bytes for RGB888)
        size_t buffer_size = result.width * result.height * 3;
        result.data.assign(buffer, buffer + buffer_size);
        free(buffer); // IMPORTANT: Free the buffer allocated by LodePNG
    } else {
        // This case should ideally not happen if error is 0, but as a safeguard:
        result.error_message = "LodePNG decode error: buffer is null without an error code.";
    }
    
    return result;
}

bool rgb888_to_png(const std::vector<uint8_t>& rgb_data,
                        unsigned width,
                        unsigned height,
                        const std::string& output_filename) {
    if (rgb_data.empty()) {
        std::cerr << "Error: RGB data is empty." << std::endl;
        return false;
    }

    // Validate that the data size matches the expected size for RGB888
    size_t expected_size = static_cast<size_t>(width) * height * 3;
    if (rgb_data.size() != expected_size) {
        std::cerr << "Error: RGB data size (" << rgb_data.size()
                  << ") does not match expected size (" << expected_size
                  << ") for width=" << width << ", height=" << height << "." << std::endl;
        return false;
    }

    // Encode the RGB888 data to a PNG file
    unsigned error = lodepng_encode24_file(output_filename.c_str(), rgb_data.data(), width, height);

    if (error) {
        std::cerr << "LodePNG encode error " << error << ": " << lodepng_error_text(error) << std::endl;
        return false;
    }

    return true;
}


/* --- Main function --- */
int main() {

    // Create the remote runner
    RemoteRunner remoteRunner(
        remotePTE.at(PTE_TYPE),
        height, width, 3,
        latHypH, latHypW, latChannels.at(PTE_TYPE).front(),
        latImgH, latImgW, latChannels.at(PTE_TYPE).back()
    );

    // Create the control runner
    ControlRunner controlRunner(
        controlPTE.at(PTE_TYPE),
        height, width, 3,
        latHypH, latHypW, latChannels.at(PTE_TYPE).front(),
        latImgH, latImgW, latChannels.at(PTE_TYPE).back()
    );

    // Load the image
    Rgb888Image image = png_to_rgb888(IMAGE_PATH);
    const std::vector<uint8_t> imageData = image.data;

    printf("Image loaded: %d x %d\n", image.width, image.height);

    rgb888_to_png(imageData, image.width, image.height, OUTPUT_PATH);

    printf("Image saved: %s\n", OUTPUT_PATH.c_str());

    // Compress the image data
    std::vector<int8_t> compressedData = remoteRunner.encodeImage(imageData);

    // Decompress the image data
    std::vector<uint8_t> decompressedData = controlRunner.decodeImage(compressedData);

    // Save the reconstructed image
    rgb888_to_png(decompressedData, image.width, image.height, OUTPUT_PATH);
    std::cout << "Image processing completed. Output saved to " << OUTPUT_PATH << std::endl;
}