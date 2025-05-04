#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <getopt.h>
#include <map>
#include <mutex>
#include <condition_variable>
#include <queue>

// Default settings
constexpr int DEFAULT_PORT = 8888;
constexpr int DEFAULT_BUFFER_SIZE = 10;

// Maximum UDP packet size (slightly less than typical MTU to avoid fragmentation)
constexpr size_t MAX_PACKET_SIZE = 1400;

// Frame header structure (must match server)
struct FrameHeader {
    uint32_t frameNumber;
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t totalSize;
    uint32_t totalPackets;
};

// Packet header structure (must match server)
struct PacketHeader {
    uint32_t frameNumber;
    uint32_t packetNumber;
    uint32_t totalPackets;
    uint32_t dataSize;
};

// Frame buffer class to store and reassemble frames
class FrameBuffer {
public:
    FrameBuffer(int maxSize) : maxBufferSize(maxSize), running(true) {}

    // Add a packet to the buffer
    void addPacket(const PacketHeader& header, const char* data, size_t dataSize) {
        std::cout << "[Line 48] Starting addPacket for frame " << header.frameNumber
                  << ", packet " << header.packetNumber << "/" << header.totalPackets << std::endl;

        std::unique_lock<std::mutex> lock(mutex);
        std::cout << "[Line 51] Acquired mutex lock in addPacket" << std::endl;

        // Get or create frame entry
        auto& frame = frames[header.frameNumber];
        std::cout << "[Line 54] Got frame entry for frame " << header.frameNumber << std::endl;

        // Initialize frame if this is the first packet
        if (frame.totalPackets == 0) {
            std::cout << "[Line 57] Initializing new frame " << header.frameNumber << std::endl;
            frame.totalPackets = header.totalPackets;
            frame.receivedPackets = 0;
            frame.data.resize(header.totalPackets);
            frame.received.resize(header.totalPackets, false);
            std::cout << "[Line 61] Frame initialized with " << header.totalPackets << " packets" << std::endl;
        }

        // Store packet data if not already received
        if (!frame.received[header.packetNumber]) {
            std::cout << "[Line 65] Storing packet " << header.packetNumber << " data (" << dataSize << " bytes)" << std::endl;
            frame.data[header.packetNumber].resize(dataSize);
            memcpy(frame.data[header.packetNumber].data(), data, dataSize);
            frame.received[header.packetNumber] = true;
            frame.receivedPackets++;
            std::cout << "[Line 70] Now have " << frame.receivedPackets << "/" << frame.totalPackets
                      << " packets for frame " << header.frameNumber << std::endl;

            // If frame is complete, add to complete frames queue
            if (frame.receivedPackets == frame.totalPackets) {
                std::cout << "[Line 74] Frame " << header.frameNumber << " is complete, adding to queue" << std::endl;
                completeFrames.push(header.frameNumber);
                condition.notify_one();
                std::cout << "[Line 77] Notified waiting threads" << std::endl;
            }
        } else {
            std::cout << "[Line 79] Packet " << header.packetNumber << " already received, skipping" << std::endl;
        }

        // Clean up old frames
        std::cout << "[Line 83] Cleaning up old frames" << std::endl;
        cleanupOldFrames();
        std::cout << "[Line 85] addPacket complete for frame " << header.frameNumber << std::endl;
    }

    // Get the next complete frame
    bool getNextFrame(cv::Mat& frame) {
        std::cout << "[Line 81] Starting getNextFrame" << std::endl;
        std::unique_lock<std::mutex> lock(mutex);
        std::cout << "[Line 83] Acquired mutex lock in getNextFrame" << std::endl;

        // Wait for a complete frame
        std::cout << "[Line 86] Waiting for a complete frame, queue size: " << completeFrames.size() << std::endl;
        while (completeFrames.empty() && running) {
            std::cout << "[Line 88] No frames available, waiting..." << std::endl;
            condition.wait_for(lock, std::chrono::milliseconds(100));
            std::cout << "[Line 90] Woke up from wait, queue size: " << completeFrames.size() << std::endl;
        }

        if (!running) {
            std::cout << "[Line 94] Not running anymore, returning false" << std::endl;
            return false;
        }

        // Get the next complete frame
        uint32_t frameNumber = completeFrames.front();
        completeFrames.pop();
        std::cout << "[Line 100] Got frame " << frameNumber << " from queue" << std::endl;

        auto& frameData = frames[frameNumber];
        std::cout << "[Line 103] Retrieved frame data for frame " << frameNumber << std::endl;

        // Calculate total size
        size_t totalSize = 0;
        for (const auto& packet : frameData.data) {
            totalSize += packet.size();
        }
        std::cout << "[Line 109] Calculated total size: " << totalSize << " bytes" << std::endl;

        // Allocate buffer for complete frame
        std::vector<char> buffer(totalSize);
        size_t offset = 0;
        std::cout << "[Line 114] Allocated buffer for frame" << std::endl;

        // Copy all packets into buffer
        std::cout << "[Line 117] Copying " << frameData.data.size() << " packets into buffer" << std::endl;
        for (const auto& packet : frameData.data) {
            memcpy(buffer.data() + offset, packet.data(), packet.size());
            offset += packet.size();
        }
        std::cout << "[Line 122] All packets copied, total bytes: " << offset << std::endl;

        // Create OpenCV Mat from buffer
        int width = frameData.width;
        int height = frameData.height;
        int channels = frameData.channels;
        std::cout << "[Line 127] Frame dimensions: " << width << "x" << height << "x" << channels << std::endl;

        if (width <= 0 || height <= 0 || channels <= 0) {
            std::cerr << "[Line 130] Invalid frame dimensions: " << width << "x" << height << "x" << channels << std::endl;
            return false;
        }

        // Create Mat with the correct dimensions
        std::cout << "[Line 135] Creating OpenCV Mat" << std::endl;
        cv::Mat receivedFrame(height, width, CV_8UC3, buffer.data());

        // Make a copy to return
        std::cout << "[Line 139] Copying Mat to output frame" << std::endl;
        receivedFrame.copyTo(frame);

        // Remove this frame from the buffer
        std::cout << "[Line 143] Removing frame " << frameNumber << " from buffer" << std::endl;
        frames.erase(frameNumber);

        std::cout << "[Line 146] getNextFrame complete, returning true" << std::endl;
        return true;
    }

    // Set frame dimensions from header
    void setFrameDimensions(uint32_t frameNumber, int width, int height, int channels) {
        std::cout << "[Line 138] Starting setFrameDimensions for frame " << frameNumber
                  << " with dimensions " << width << "x" << height << "x" << channels << std::endl;

        std::unique_lock<std::mutex> lock(mutex);
        std::cout << "[Line 142] Acquired mutex lock in setFrameDimensions" << std::endl;

        auto& frame = frames[frameNumber];
        frame.width = width;
        frame.height = height;
        frame.channels = channels;

        std::cout << "[Line 148] Frame dimensions set for frame " << frameNumber << std::endl;
    }

    // Stop the buffer
    void stop() {
        std::cout << "[Line 147] Starting stop" << std::endl;
        std::unique_lock<std::mutex> lock(mutex);
        std::cout << "[Line 149] Acquired mutex lock in stop" << std::endl;

        running = false;
        condition.notify_all();

        std::cout << "[Line 153] Notified all waiting threads, buffer stopped" << std::endl;
    }

private:
    struct Frame {
        int width = 0;
        int height = 0;
        int channels = 0;
        uint32_t totalPackets = 0;
        uint32_t receivedPackets = 0;
        std::vector<std::vector<char>> data;
        std::vector<bool> received;
    };

    void cleanupOldFrames() {
        std::cout << "[Line 165] Starting cleanupOldFrames, current frame count: " << frames.size() << std::endl;

        // Keep only the newest maxBufferSize frames
        if (frames.size() > maxBufferSize) {
            std::cout << "[Line 169] Buffer size exceeded, cleaning up old frames" << std::endl;
            uint32_t oldestFrame = frames.begin()->first;
            uint32_t threshold = oldestFrame + maxBufferSize;
            std::cout << "[Line 172] Threshold for cleanup: frame " << threshold << std::endl;

            int removedCount = 0;
            auto it = frames.begin();
            while (it != frames.end()) {
                if (it->first < threshold) {
                    std::cout << "[Line 177] Removing frame " << it->first << std::endl;
                    it = frames.erase(it);
                    removedCount++;
                } else {
                    ++it;
                }
            }
            std::cout << "[Line 183] Cleanup complete, removed " << removedCount << " frames" << std::endl;
        } else {
            std::cout << "[Line 185] No cleanup needed, buffer size within limits" << std::endl;
        }
    }

    std::mutex mutex;
    std::condition_variable condition;
    std::map<uint32_t, Frame> frames;
    std::queue<uint32_t> completeFrames;
    int maxBufferSize;
    std::atomic<bool> running;
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "Options:\n"
              << "  --port PORT       Port to listen on (default: " << DEFAULT_PORT << ")\n"
              << "  --server IP       Server IP address (required)\n"
              << "  --buffer SIZE     Frame buffer size (default: " << DEFAULT_BUFFER_SIZE << ")\n"
              << "  --help            Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::cout << "[Line 198] Starting UDP video client" << std::endl;

    // Parse command line arguments
    int port = DEFAULT_PORT;
    std::string serverIP = "";
    int bufferSize = DEFAULT_BUFFER_SIZE;

    std::cout << "[Line 204] Parsing command line arguments" << std::endl;
    const struct option long_options[] = {
        {"port", required_argument, nullptr, 'p'},
        {"server", required_argument, nullptr, 's'},
        {"buffer", required_argument, nullptr, 'b'},
        {"help", no_argument, nullptr, '?'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "p:s:b:?", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'p':
                port = std::stoi(optarg);
                std::cout << "[Line 217] Set port to " << port << std::endl;
                break;
            case 's':
                serverIP = optarg;
                std::cout << "[Line 221] Set server IP to " << serverIP << std::endl;
                break;
            case 'b':
                bufferSize = std::stoi(optarg);
                std::cout << "[Line 225] Set buffer size to " << bufferSize << std::endl;
                break;
            case '?':
                std::cout << "[Line 228] Showing usage and exiting" << std::endl;
                printUsage(argv[0]);
                return 0;
            default:
                std::cout << "[Line 232] Invalid option, showing usage and exiting" << std::endl;
                printUsage(argv[0]);
                return 1;
        }
    }

    if (serverIP.empty()) {
        std::cerr << "[Line 239] Error: Server IP address is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "[Line 244] Command line arguments parsed successfully" << std::endl;

    // Create UDP socket
    std::cout << "[Line 233] Creating UDP socket" << std::endl;
    int clientSocket = socket(AF_INET, SOCK_DGRAM, 0);
    if (clientSocket < 0) {
        std::cerr << "[Line 236] Error creating socket: " << strerror(errno) << std::endl;
        return 1;
    }
    std::cout << "[Line 239] UDP socket created successfully" << std::endl;

    // Set socket options to allow reuse of address
    std::cout << "[Line 242] Setting socket options (SO_REUSEADDR)" << std::endl;
    int opt_val = 1;
    if (setsockopt(clientSocket, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val)) < 0) {
        std::cerr << "[Line 245] Error setting socket options: " << strerror(errno) << std::endl;
        close(clientSocket);
        return 1;
    }
    std::cout << "[Line 249] Socket options set successfully" << std::endl;

    // Increase receive buffer size
    std::cout << "[Line 252] Setting receive buffer size to 8MB" << std::endl;
    int recvBufSize = 8 * 1024 * 1024; // 8 MB
    if (setsockopt(clientSocket, SOL_SOCKET, SO_RCVBUF, &recvBufSize, sizeof(recvBufSize)) < 0) {
        std::cerr << "[Line 255] Warning: Could not set receive buffer size: " << strerror(errno) << std::endl;
    } else {
        std::cout << "[Line 257] Receive buffer size set successfully" << std::endl;
    }

    // Try to disable UDP checksums for better performance
    #ifdef __linux__
    std::cout << "[Line 262] Attempting to disable UDP checksums" << std::endl;
    int val = 1;
    if (setsockopt(clientSocket, SOL_SOCKET, SO_NO_CHECK, &val, sizeof(val)) < 0) {
        std::cerr << "[Line 265] Warning: Could not disable UDP checksums: " << strerror(errno) << std::endl;
    } else {
        std::cout << "[Line 267] UDP checksums disabled successfully" << std::endl;
    }
    #endif

    // Bind socket to port
    std::cout << "[Line 272] Binding socket to port " << port << std::endl;
    struct sockaddr_in clientAddr;
    memset(&clientAddr, 0, sizeof(clientAddr));
    clientAddr.sin_family = AF_INET;
    clientAddr.sin_addr.s_addr = INADDR_ANY;
    clientAddr.sin_port = htons(port);

    if (bind(clientSocket, (struct sockaddr*)&clientAddr, sizeof(clientAddr)) < 0) {
        std::cerr << "[Line 280] Error binding socket: " << strerror(errno) << std::endl;
        close(clientSocket);
        return 1;
    }
    std::cout << "[Line 284] Socket bound successfully" << std::endl;

    // Set up server address
    std::cout << "[Line 287] Setting up server address: " << serverIP << ":" << port << std::endl;
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    if (inet_pton(AF_INET, serverIP.c_str(), &serverAddr.sin_addr) <= 0) {
        std::cerr << "[Line 293] Invalid server address: " << serverIP << std::endl;
        close(clientSocket);
        return 1;
    }
    std::cout << "[Line 297] Server address set up successfully" << std::endl;

    std::cout << "[Line 299] UDP Video Client started on port " << port << std::endl;
    std::cout << "[Line 300] Connecting to server at " << serverIP << ":" << port << std::endl;

    // Send initial packet to server to establish connection
    std::cout << "[Line 303] Sending initial packet to server" << std::endl;
    const char* initMsg = "INIT";
    if (sendto(clientSocket, initMsg, strlen(initMsg), 0,
              (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "[Line 307] Error sending initial packet: " << strerror(errno) << std::endl;
        close(clientSocket);
        return 1;
    }
    std::cout << "[Line 311] Initial packet sent successfully" << std::endl;

    // Create frame buffer
    std::cout << "[Line 300] Creating frame buffer with size " << bufferSize << std::endl;
    FrameBuffer frameBuffer(bufferSize);
    std::cout << "[Line 302] Frame buffer created successfully" << std::endl;

    // Create window for display
    std::cout << "[Line 305] Creating display window" << std::endl;
    cv::namedWindow("UDP Video Stream", cv::WINDOW_NORMAL);
    std::cout << "[Line 307] Display window created successfully" << std::endl;

    // Variables for FPS calculation
    std::cout << "[Line 310] Initializing FPS calculation variables" << std::endl;
    int frameCount = 0;
    auto fpsStartTime = std::chrono::steady_clock::now();
    double fps = 0.0;

    // Variables for data rate calculation
    std::cout << "[Line 315] Initializing data rate calculation variables" << std::endl;
    size_t totalBytesReceived = 0;
    auto dataRateStartTime = std::chrono::steady_clock::now();
    double dataRateMbps = 0.0;

    // Start receiver thread
    std::cout << "[Line 320] Starting receiver thread" << std::endl;
    std::atomic<bool> running(true);
    std::thread receiverThread([&]() {
        std::cout << "[Line 323] Receiver thread started" << std::endl;
        char buffer[MAX_PACKET_SIZE];
        struct sockaddr_in senderAddr;
        socklen_t senderAddrLen = sizeof(senderAddr);

        std::cout << "[Line 328] Entering receiver loop" << std::endl;
        int packetCount = 0;
        int headerCount = 0;
        int dataCount = 0;

        while (running) {
            // Receive packet
            if (packetCount % 1000 == 0) {
                std::cout << "[Line 335] Waiting for packet #" << packetCount << std::endl;
            }

            ssize_t received = recvfrom(clientSocket, buffer, MAX_PACKET_SIZE, 0,
                                       (struct sockaddr*)&senderAddr, &senderAddrLen);

            if (received <= 0) {
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    std::cerr << "[Line 342] Error receiving packet: " << strerror(errno) << std::endl;
                    break;
                }
                continue;
            }

            packetCount++;
            if (packetCount % 1000 == 0) {
                std::cout << "[Line 349] Received packet #" << packetCount
                          << " (" << received << " bytes)" << std::endl;
            }

            // Update data rate statistics
            totalBytesReceived += received;

            // Check if this is a frame header packet
            if (received == sizeof(FrameHeader)) {
                FrameHeader* header = reinterpret_cast<FrameHeader*>(buffer);
                headerCount++;

                if (headerCount % 30 == 0) {
                    std::cout << "[Line 361] Received frame header #" << headerCount
                              << " for frame " << header->frameNumber
                              << " (" << header->width << "x" << header->height << ")" << std::endl;
                }

                frameBuffer.setFrameDimensions(header->frameNumber, header->width, header->height, header->channels);
            }
            // Check if this is a data packet
            else if (received >= sizeof(PacketHeader)) {
                PacketHeader* header = reinterpret_cast<PacketHeader*>(buffer);
                const char* data = buffer + sizeof(PacketHeader);
                size_t dataSize = received - sizeof(PacketHeader);

                dataCount++;
                if (dataCount % 1000 == 0) {
                    std::cout << "[Line 374] Received data packet #" << dataCount
                              << " for frame " << header->frameNumber
                              << " (packet " << header->packetNumber << "/" << header->totalPackets << ")" << std::endl;
                }

                frameBuffer.addPacket(*header, data, dataSize);
            }
        }

        std::cout << "[Line 383] Receiver thread exiting" << std::endl;
    });

    std::cout << "[Line 386] Receiver thread started successfully" << std::endl;

    // Main display loop
    std::cout << "[Line 355] Starting main display loop" << std::endl;
    try {
        cv::Mat frame;
        int displayedFrames = 0;

        std::cout << "[Line 360] Entering display loop" << std::endl;
        while (running) {
            // Get next complete frame
            std::cout << "[Line 363] Attempting to get next frame" << std::endl;
            if (frameBuffer.getNextFrame(frame)) {
                displayedFrames++;
                std::cout << "[Line 366] Got frame #" << displayedFrames
                          << " with size " << frame.cols << "x" << frame.rows << std::endl;

                // Update FPS counter
                frameCount++;
                auto now = std::chrono::steady_clock::now();
                auto fpsElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - fpsStartTime).count();
                if (fpsElapsed >= 1) {
                    fps = frameCount / static_cast<double>(fpsElapsed);
                    frameCount = 0;
                    fpsStartTime = now;
                    std::cout << "[Line 376] FPS updated: " << fps << std::endl;
                }

                // Update data rate
                auto dataRateElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - dataRateStartTime).count();
                if (dataRateElapsed >= 1) {
                    dataRateMbps = (totalBytesReceived * 8.0 / 1000000.0) / dataRateElapsed;
                    std::cout << "[Line 383] Data rate updated: " << dataRateMbps << " Mbps" << std::endl;
                    totalBytesReceived = 0;
                    dataRateStartTime = now;
                }

                // Display stats on frame
                std::string statsText = cv::format("Resolution: %dx%d | FPS: %.1f | Data Rate: %.2f Mbps",
                                                  frame.cols, frame.rows, fps, dataRateMbps);
                std::cout << "[Line 390] Adding stats to frame: " << statsText << std::endl;
                cv::putText(frame, statsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                           cv::Scalar(0, 255, 0), 2);

                // Display frame
                std::cout << "[Line 394] Displaying frame #" << displayedFrames << std::endl;
                cv::imshow("UDP Video Stream", frame);
                std::cout << "[Line 396] Frame displayed successfully" << std::endl;
            } else {
                std::cout << "[Line 398] No frame available yet" << std::endl;
            }

            // Check for key press (ESC to exit)
            int key = cv::waitKey(1);
            if (key == 27) {  // ESC key
                std::cout << "[Line 403] ESC key pressed, exiting" << std::endl;
                running = false;
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Line 409] Exception in main display loop: " << e.what() << std::endl;
    }

    // Clean up
    std::cout << "[Line 413] Starting cleanup" << std::endl;
    running = false;
    std::cout << "[Line 415] Stopping frame buffer" << std::endl;
    frameBuffer.stop();
    std::cout << "[Line 417] Waiting for receiver thread to join" << std::endl;
    receiverThread.join();
    std::cout << "[Line 419] Closing socket" << std::endl;
    close(clientSocket);
    std::cout << "[Line 421] Destroying windows" << std::endl;
    cv::destroyAllWindows();
    std::cout << "[Line 423] Client stopped" << std::endl;

    return 0;
}
