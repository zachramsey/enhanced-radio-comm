/* remote_runner.h */
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

// Forward declaration of the implementation class/struct
struct RemoteRunnerImpl;

/**
 * RemoteRunner class for handling video model operations
 * Provides functionality to decompress latent data
 */
class RemoteRunner {
private:
    // Pointer to the private implementation
    std::unique_ptr<RemoteRunnerImpl> remoteRunnerImpl;

public:
    /**
     * Initialize the RemoteRunner
     * @param modelPath Path to the exported video model (.pte file)
     * @param imgH      Height of the input image
     * @param imgW      Width of the input image
     * @param imgC      Number of channels in the input image
     * @param latHypH   Height of the latent hyper
     * @param latHypW   Width of the latent hyper
     * @param latHypC   Number of channels in the latent hyper
     * @param latImgH   Height of the latent image
     * @param latImgW   Width of the latent image
     * @param latImgC   Number of channels in the latent image
     */
    RemoteRunner(
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
    );

    // Destructor
    ~RemoteRunner();

    // Move constructor
    RemoteRunner(RemoteRunner&&) noexcept;

    // Move assignment operator
    RemoteRunner& operator=(RemoteRunner&&) noexcept;

    // Delete copy semantics (unique_ptr is not copyable)
    RemoteRunner(const RemoteRunner&) = delete;
    RemoteRunner& operator=(const RemoteRunner&) = delete;

    /**
     * Compress image data using the video encoder model.
     * @param data Pointer to the input image data (RGB888 format).
     * @return A vector containing the compressed image data. Vector might be empty on failure.
     */
    std::vector<uint8_t> encodeImage(const std::vector<uint8_t>& data) const;
};
