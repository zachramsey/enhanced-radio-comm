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
     * @param videoModelPath Path to the exported video model (.pte file)
     * @param latImageHeight Height of the latent image
     * @param latImageWidth Width of the latent image
     * @param latImageChannel Number of channels in the latent image
     * @param latHyperHeight Height of the latent hyper
     * @param latHyperWidth Width of the latent hyper
     * @param latHyperChannel Number of channels in the latent hyper
     * @throws std::runtime_error if model loading or initialization fails.
     */
    ControlRunner(
        const std::string& videoModelPath,
        int latImageHeight,
        int latImageWidth,
        int latImageChannel,
        int latHyperHeight,
        int latHyperWidth,
        int latHyperChannel
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
    std::vector<uint8_t> decodeImage(const uint8_t* data);
};
