
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <list>
#include <png.h>

#include <remote_runner.h>
#include <control_runner.h>

const std::map<std::string, std::string> remotePTE = {
    {"small_q8", "../pte/remote/img_enc_small_xnnpack_q8.pte"},
    {"small_f32", "../pte/remote/img_enc_small_xnnpack_f32.pte"},
    {"default_q8", "../pte/remote/img_enc_default_xnnpack_q8.pte"},
    {"default_f32", "../pte/remote/img_enc_default_xnnpack_f32.pte"}
};

const std::map<std::string, std::string> controlPTE = {
    {"small_q8", "../pte/control/img_enc_small_xnnpack_q8.pte"},
    {"small_f32", "../pte/control/img_enc_small_xnnpack_f32.pte"},
    {"default_q8", "../pte/control/img_enc_default_xnnpack_q8.pte"},
    {"default_f32", "../pte/control/img_enc_default_xnnpack_f32.pte"}
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

const std::string IMAGE_PATH = "test_image.png";
const std::string OUTPUT_PATH = "output_image.png";
const std::string PTE_TYPE = "small_f32";


/* --- Load a PNG image and convert it to RGB888 format --- */
std::vector<uint8_t> png_to_rgb888(const std::string& filename, int& width, int& height) {
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        fclose(fp);
        std::cerr << "Error: png_create_read_struct failed" << std::endl;
        return {};
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fclose(fp);
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        std::cerr << "Error: png_create_info_struct failed" << std::endl;
        return {};
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fclose(fp);
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        std::cerr << "Error: Error during png read" << std::endl;
        return {};
    }

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    if (bit_depth == 16)
        png_set_strip_16(png_ptr);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);

    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);

    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);

    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    int rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    std::vector<uint8_t> image_data(width * height * 3);
    std::vector<png_bytep> row_pointers(height);

    for (int i = 0; i < height; ++i) {
        row_pointers[i] = &image_data[i * rowbytes];
    }

    png_read_image(png_ptr, row_pointers.data());
    png_read_end(png_ptr, info_ptr);

    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(fp);

    // Extract RGB data, discarding alpha channel if present
    std::vector<uint8_t> rgb_data;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            rgb_data.push_back(image_data[i * rowbytes + j * 4]);     // R
            rgb_data.push_back(image_data[i * rowbytes + j * 4 + 1]); // G
            rgb_data.push_back(image_data[i * rowbytes + j * 4 + 2]); // B
        }
    }

    return rgb_data;
}


/* --- Convert RGB888 data to PNG format and save it --- */
void rgb888_to_png(const std::string& filename, const std::vector<uint8_t>& rgb_data, int width, int height) {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        fclose(fp);
        std::cerr << "Error: png_create_write_struct failed" << std::endl;
        return;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fclose(fp);
        png_destroy_write_struct(&png_ptr, nullptr);
        std::cerr << "Error: png_create_info_struct failed" << std::endl;
        return;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fclose(fp);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        std::cerr << "Error: Error during png write" << std::endl;
        return;
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    std::vector<png_bytep> row_pointers(height);
    for (int i = 0; i < height; ++i) {
        row_pointers[i] = (png_bytep)&rgb_data[i * width * 3];
    }

    png_write_image(png_ptr, row_pointers.data());
    png_write_end(png_ptr, nullptr);

    fclose(fp);
    png_destroy_write_struct(&png_ptr, &info_ptr);
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
    const std::vector<uint8_t> imageData = png_to_rgb888(IMAGE_PATH, width, height);

    printf("Image loaded: %d x %d\n", width, height);

    rgb888_to_png(OUTPUT_PATH, imageData, width, height);

    printf("Image saved: %s\n", OUTPUT_PATH.c_str());

    // Compress the image data
    std::vector<int8_t> compressedData = remoteRunner.encodeImage(imageData);

    // Decompress the image data
    std::vector<uint8_t> decompressedData = controlRunner.decodeImage(compressedData);

    // Save the reconstructed image
    rgb888_to_png(OUTPUT_PATH, decompressedData, width, height);
    std::cout << "Image processing completed. Output saved to " << OUTPUT_PATH << std::endl;
}