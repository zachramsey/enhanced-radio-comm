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
     * @param videoModelPath Path to the exported video model (.pte file)
     * @param imageHeight Height of the input image
     * @param imageWidth Width of the input image
     * @param imageChannel Number of channels in the input image
     * @throws std::runtime_error if model loading or initialization fails.
     */
    RemoteRunner(
        const std::string& videoModelPath,
        int imageHeight,
        int imageWidth,
        int imageChannel
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
     * @param data Pointer to input data (image).
     * The size is assumed to be imgSize bytes.
     * @return A vector containing the compressed image data. Vector might be empty on failure.
     */
    std::vector<uint8_t> encodeImage(const uint8_t* data);
};
