/* control_runner.h */
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

// Forward declaration of the implementation
struct ControlRunnerImpl;

/**
 * ControlRunner class for handling video model operations
 * Provides functionality to decompress latent data
 */
class ControlRunner {
private:
    // Pointer to the private implementation
    std::unique_ptr<ControlRunnerImpl> controlRunnerImpl;

public:
    /**
     * Initialize the ControlRunner
     * @param modelPath Path to the exported video model (.pte file)
     * @param imgH Height of the image output
     * @param imgW Width of the image output
     * @param imgC Number of channels in the image output
     * @param latHypH Height of the latent hyper input
     * @param latHypW Width of the latent hyper input
     * @param latHypC Number of channels in the latent hyper input
     * @param latImgH Height of the latent image input
     * @param latImgW Width of the latent image input
     * @param latImgC Number of channels in the latent image input
     */
    ControlRunner(
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
    );

    // Destructor
    ~ControlRunner();

    // Move constructor
    ControlRunner(ControlRunner&&) noexcept;

    // Move assignment operator
    ControlRunner& operator=(ControlRunner&&) noexcept;

    // Delete copy semantics (unique_ptr is not copyable)
    ControlRunner(const ControlRunner&) = delete;
    ControlRunner& operator=(const ControlRunner&) = delete;

    /**
     * Decompress latent data using the video model.
     * @param data Pointer to compressed input data (containing both hyper and image latent parts).
     * The size is assumed to be latHyperSize + latImgSize bytes.
     * @return A vector containing the decompressed image data. Vector might be empty on failure.
     */
    std::vector<uint8_t> decodeImage(const std::vector<uint8_t>& data);
};
