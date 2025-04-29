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
        std::unique_lock<std::mutex> lock(mutex);

        // Get or create frame entry
        auto& frame = frames[header.frameNumber];

        // Initialize frame if this is the first packet
        if (frame.totalPackets == 0) {
            frame.totalPackets = header.totalPackets;
            frame.receivedPackets = 0;
            frame.data.resize(header.totalPackets);
            frame.received.resize(header.totalPackets, false);
        }

        // Store packet data if not already received
        if (!frame.received[header.packetNumber]) {
            frame.data[header.packetNumber].resize(dataSize);
            memcpy(frame.data[header.packetNumber].data(), data, dataSize);
            frame.received[header.packetNumber] = true;
            frame.receivedPackets++;

            // If frame is complete, add to complete frames queue
            if (frame.receivedPackets == frame.totalPackets) {
                completeFrames.push(header.frameNumber);
                condition.notify_one();
            }
        }

        // Clean up old frames
        cleanupOldFrames();
    }

    // Get the next complete frame
    bool getNextFrame(cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mutex);

        // Wait for a complete frame
        while (completeFrames.empty() && running) {
            condition.wait_for(lock, std::chrono::milliseconds(100));
        }

        if (!running) {
            return false;
        }

        // Get the next complete frame
        uint32_t frameNumber = completeFrames.front();
        completeFrames.pop();

        auto& frameData = frames[frameNumber];

        // Calculate total size
        size_t totalSize = 0;
        for (const auto& packet : frameData.data) {
            totalSize += packet.size();
        }

        // Allocate buffer for complete frame
        std::vector<char> buffer(totalSize);
        size_t offset = 0;

        // Copy all packets into buffer
        for (const auto& packet : frameData.data) {
            memcpy(buffer.data() + offset, packet.data(), packet.size());
            offset += packet.size();
        }

        // Create OpenCV Mat from buffer
        int width = frameData.width;
        int height = frameData.height;
        int channels = frameData.channels;

        if (width <= 0 || height <= 0 || channels <= 0) {
            std::cerr << "Invalid frame dimensions: " << width << "x" << height << "x" << channels << std::endl;
            return false;
        }

        // Create Mat with the correct dimensions
        cv::Mat receivedFrame(height, width, CV_8UC3, buffer.data());

        // Make a copy to return
        receivedFrame.copyTo(frame);

        // Remove this frame from the buffer
        frames.erase(frameNumber);

        return true;
    }

    // Set frame dimensions from header
    void setFrameDimensions(uint32_t frameNumber, int width, int height, int channels) {
        std::unique_lock<std::mutex> lock(mutex);
        auto& frame = frames[frameNumber];
        frame.width = width;
        frame.height = height;
        frame.channels = channels;
    }

    // Stop the buffer
    void stop() {
        std::unique_lock<std::mutex> lock(mutex);
        running = false;
        condition.notify_all();
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
        // Keep only the newest maxBufferSize frames
        if (frames.size() > maxBufferSize) {
            uint32_t oldestFrame = frames.begin()->first;
            uint32_t threshold = oldestFrame + maxBufferSize;

            auto it = frames.begin();
            while (it != frames.end()) {
                if (it->first < threshold) {
                    it = frames.erase(it);
                } else {
                    ++it;
                }
            }
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
    // Parse command line arguments
    int port = DEFAULT_PORT;
    std::string serverIP = "";
    int bufferSize = DEFAULT_BUFFER_SIZE;

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
            case 'p': port = std::stoi(optarg); break;
            case 's': serverIP = optarg; break;
            case 'b': bufferSize = std::stoi(optarg); break;
            case '?':
                printUsage(argv[0]);
                return 0;
            default:
                printUsage(argv[0]);
                return 1;
        }
    }

    if (serverIP.empty()) {
        std::cerr << "Error: Server IP address is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Create UDP socket
    int clientSocket = socket(AF_INET, SOCK_DGRAM, 0);
    if (clientSocket < 0) {
        std::cerr << "Error creating socket: " << strerror(errno) << std::endl;
        return 1;
    }

    // Set socket options to allow reuse of address
    int opt_val = 1;
    if (setsockopt(clientSocket, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val)) < 0) {
        std::cerr << "Error setting socket options: " << strerror(errno) << std::endl;
        close(clientSocket);
        return 1;
    }

    // Increase receive buffer size
    int recvBufSize = 8 * 1024 * 1024; // 8 MB
    if (setsockopt(clientSocket, SOL_SOCKET, SO_RCVBUF, &recvBufSize, sizeof(recvBufSize)) < 0) {
        std::cerr << "Warning: Could not set receive buffer size: " << strerror(errno) << std::endl;
    }

    // Bind socket to port
    struct sockaddr_in clientAddr;
    memset(&clientAddr, 0, sizeof(clientAddr));
    clientAddr.sin_family = AF_INET;
    clientAddr.sin_addr.s_addr = INADDR_ANY;
    clientAddr.sin_port = htons(port);

    if (bind(clientSocket, (struct sockaddr*)&clientAddr, sizeof(clientAddr)) < 0) {
        std::cerr << "Error binding socket: " << strerror(errno) << std::endl;
        close(clientSocket);
        return 1;
    }

    // Set up server address
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    if (inet_pton(AF_INET, serverIP.c_str(), &serverAddr.sin_addr) <= 0) {
        std::cerr << "Invalid server address: " << serverIP << std::endl;
        close(clientSocket);
        return 1;
    }

    std::cout << "UDP Video Client started on port " << port << std::endl;
    std::cout << "Connecting to server at " << serverIP << ":" << port << std::endl;

    // Send initial packet to server to establish connection
    const char* initMsg = "INIT";
    if (sendto(clientSocket, initMsg, strlen(initMsg), 0,
              (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Error sending initial packet: " << strerror(errno) << std::endl;
        close(clientSocket);
        return 1;
    }

    // Create frame buffer
    FrameBuffer frameBuffer(bufferSize);

    // Create window for display
    cv::namedWindow("UDP Video Stream", cv::WINDOW_NORMAL);

    // Variables for FPS calculation
    int frameCount = 0;
    auto fpsStartTime = std::chrono::steady_clock::now();
    double fps = 0.0;

    // Variables for data rate calculation
    size_t totalBytesReceived = 0;
    auto dataRateStartTime = std::chrono::steady_clock::now();
    double dataRateMbps = 0.0;

    // Start receiver thread
    std::atomic<bool> running(true);
    std::thread receiverThread([&]() {
        char buffer[MAX_PACKET_SIZE];
        struct sockaddr_in senderAddr;
        socklen_t senderAddrLen = sizeof(senderAddr);

        while (running) {
            // Receive packet
            ssize_t received = recvfrom(clientSocket, buffer, MAX_PACKET_SIZE, 0,
                                       (struct sockaddr*)&senderAddr, &senderAddrLen);

            if (received <= 0) {
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    std::cerr << "Error receiving packet: " << strerror(errno) << std::endl;
                    break;
                }
                continue;
            }

            // Update data rate statistics
            totalBytesReceived += received;

            // Check if this is a frame header packet
            if (received == sizeof(FrameHeader)) {
                FrameHeader* header = reinterpret_cast<FrameHeader*>(buffer);
                frameBuffer.setFrameDimensions(header->frameNumber, header->width, header->height, header->channels);
            }
            // Check if this is a data packet
            else if (received >= sizeof(PacketHeader)) {
                PacketHeader* header = reinterpret_cast<PacketHeader*>(buffer);
                const char* data = buffer + sizeof(PacketHeader);
                size_t dataSize = received - sizeof(PacketHeader);

                frameBuffer.addPacket(*header, data, dataSize);
            }
        }
    });

    // Main display loop
    try {
        cv::Mat frame;

        while (running) {
            // Get next complete frame
            if (frameBuffer.getNextFrame(frame)) {
                // Update FPS counter
                frameCount++;
                auto now = std::chrono::steady_clock::now();
                auto fpsElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - fpsStartTime).count();
                if (fpsElapsed >= 1) {
                    fps = frameCount / static_cast<double>(fpsElapsed);
                    frameCount = 0;
                    fpsStartTime = now;
                }

                // Update data rate
                auto dataRateElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - dataRateStartTime).count();
                if (dataRateElapsed >= 1) {
                    dataRateMbps = (totalBytesReceived * 8.0 / 1000000.0) / dataRateElapsed;
                    totalBytesReceived = 0;
                    dataRateStartTime = now;
                }

                // Display stats on frame
                std::string statsText = cv::format("Resolution: %dx%d | FPS: %.1f | Data Rate: %.2f Mbps",
                                                  frame.cols, frame.rows, fps, dataRateMbps);
                cv::putText(frame, statsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                           cv::Scalar(0, 255, 0), 2);

                // Display frame
                cv::imshow("UDP Video Stream", frame);
            }

            // Check for key press (ESC to exit)
            int key = cv::waitKey(1);
            if (key == 27) {  // ESC key
                running = false;
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    // Clean up
    running = false;
    frameBuffer.stop();
    receiverThread.join();
    close(clientSocket);
    cv::destroyAllWindows();
    std::cout << "Client stopped" << std::endl;

    return 0;
}
