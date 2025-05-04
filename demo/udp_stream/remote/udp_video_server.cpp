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

// Default settings
constexpr int DEFAULT_PORT = 8888;
constexpr int DEFAULT_CAMERA_INDEX = 1;
constexpr int DEFAULT_WIDTH = 640;
constexpr int DEFAULT_HEIGHT = 480;
constexpr int DEFAULT_FPS = 30;

// Maximum UDP packet size (increased to reduce number of packets)
constexpr size_t MAX_PACKET_SIZE = 65000;

// Frame header structure
struct FrameHeader {
    uint32_t frameNumber;
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t totalSize;
    uint32_t totalPackets;
};

// Packet header structure
struct PacketHeader {
    uint32_t frameNumber;
    uint32_t packetNumber;
    uint32_t totalPackets;
    uint32_t dataSize;
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "Options:\n"
              << "  --port PORT       Port to listen on (default: " << DEFAULT_PORT << ")\n"
              << "  --camera INDEX    Camera device index (default: " << DEFAULT_CAMERA_INDEX << ")\n"
              << "  --width WIDTH     Frame width (default: " << DEFAULT_WIDTH << ")\n"
              << "  --height HEIGHT   Frame height (default: " << DEFAULT_HEIGHT << ")\n"
              << "  --fps FPS         Target frames per second (default: " << DEFAULT_FPS << ")\n"
              << "  --client IP       Client IP address (if not specified, waits for first packet)\n"
              << "  --help            Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int port = DEFAULT_PORT;
    int cameraIndex = DEFAULT_CAMERA_INDEX;
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int fps = DEFAULT_FPS;
    std::string clientIP = "";

    const struct option long_options[] = {
        {"port", required_argument, nullptr, 'p'},
        {"camera", required_argument, nullptr, 'c'},
        {"width", required_argument, nullptr, 'w'},
        {"height", required_argument, nullptr, 'h'},
        {"fps", required_argument, nullptr, 'f'},
        {"client", required_argument, nullptr, 'a'},
        {"help", no_argument, nullptr, '?'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "p:c:w:h:f:a:?", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'p': port = std::stoi(optarg); break;
            case 'c': cameraIndex = std::stoi(optarg); break;
            case 'w': width = std::stoi(optarg); break;
            case 'h': height = std::stoi(optarg); break;
            case 'f': fps = std::stoi(optarg); break;
            case 'a': clientIP = optarg; break;
            case '?':
                printUsage(argv[0]);
                return 0;
            default:
                printUsage(argv[0]);
                return 1;
        }
    }

    // Create UDP socket
    int serverSocket = socket(AF_INET, SOCK_DGRAM, 0);
    if (serverSocket < 0) {
        std::cerr << "Error creating socket: " << strerror(errno) << std::endl;
        return 1;
    }

    // Set socket options to allow reuse of address
    int opt_val = 1;
    if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val)) < 0) {
        std::cerr << "Error setting socket options: " << strerror(errno) << std::endl;
        close(serverSocket);
        return 1;
    }

    // Increase send buffer size for larger packets
    int sendBufSize = 16 * 1024 * 1024; // 16 MB
    if (setsockopt(serverSocket, SOL_SOCKET, SO_SNDBUF, &sendBufSize, sizeof(sendBufSize)) < 0) {
        std::cerr << "Warning: Could not set send buffer size: " << strerror(errno) << std::endl;
    } else {
        std::cout << "Send buffer size set to " << (sendBufSize / 1024 / 1024) << " MB" << std::endl;
    }

    // Try to disable UDP checksums for better performance
    #ifdef __linux__
    int val = 1;
    if (setsockopt(serverSocket, SOL_SOCKET, SO_NO_CHECK, &val, sizeof(val)) < 0) {
        std::cerr << "Warning: Could not disable UDP checksums: " << strerror(errno) << std::endl;
    } else {
        std::cout << "UDP checksums disabled" << std::endl;
    }
    #endif

    // Bind socket to port
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(port);

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Error binding socket: " << strerror(errno) << std::endl;
        close(serverSocket);
        return 1;
    }

    std::cout << "UDP Video Server started on port " << port << std::endl;

    // Client address
    struct sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    bool clientConnected = false;

    // If client IP is specified, use it
    if (!clientIP.empty()) {
        clientAddr.sin_family = AF_INET;
        clientAddr.sin_port = htons(port);
        if (inet_pton(AF_INET, clientIP.c_str(), &clientAddr.sin_addr) <= 0) {
            std::cerr << "Invalid client address: " << clientIP << std::endl;
            close(serverSocket);
            return 1;
        }
        clientConnected = true;
        std::cout << "Using specified client IP: " << clientIP << std::endl;
    } else {
        // Wait for a packet from client to get its address
        std::cout << "Waiting for client to send a packet..." << std::endl;
        char buffer[MAX_PACKET_SIZE];
        ssize_t received = recvfrom(serverSocket, buffer, MAX_PACKET_SIZE, 0,
                                   (struct sockaddr*)&clientAddr, &clientAddrLen);
        if (received < 0) {
            std::cerr << "Error receiving initial packet: " << strerror(errno) << std::endl;
            close(serverSocket);
            return 1;
        }
        clientConnected = true;
        std::cout << "Client connected: " << inet_ntoa(clientAddr.sin_addr) << std::endl;
    }

    // Open camera
    cv::VideoCapture cap;
    bool usingTestPattern = false;

    // Try to open camera with different backends
    std::vector<int> backends = {
        cv::CAP_ANY,       // Auto-detect
        cv::CAP_V4L2,      // Video for Linux
        cv::CAP_GSTREAMER, // GStreamer
        cv::CAP_FFMPEG     // FFmpeg
    };

    for (int backend : backends) {
        std::cout << "Trying to open camera " << cameraIndex << " with backend " << backend << "..." << std::endl;
        cap.open(cameraIndex, backend);
        if (cap.isOpened()) {
            std::cout << "Successfully opened camera with backend " << backend << std::endl;
            break;
        }
    }

    if (!cap.isOpened()) {
        std::cerr << "Warning: Could not open camera. Using test pattern instead." << std::endl;
        usingTestPattern = true;
    } else {
        // Set camera properties
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap.set(cv::CAP_PROP_FPS, fps);

        // Get actual camera properties
        width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        // Verify that we got valid dimensions
        if (width <= 0 || height <= 0) {
            std::cerr << "Warning: Invalid camera resolution: " << width << "x" << height << std::endl;
            std::cerr << "Using test pattern with default resolution instead." << std::endl;
            cap.release();
            usingTestPattern = true;
            width = DEFAULT_WIDTH;
            height = DEFAULT_HEIGHT;
        } else {
            std::cout << "Camera opened. Actual resolution: " << width << "x" << height << std::endl;
        }
    }

    // Calculate frame interval for FPS control
    std::chrono::microseconds frameInterval(1000000 / fps);

    // Main streaming loop
    try {
        cv::Mat frame;
        uint32_t frameNumber = 0;

        while (true) {
            auto frameStart = std::chrono::steady_clock::now();

            if (usingTestPattern) {
                // Generate test pattern
                frame = cv::Mat(height, width, CV_8UC3);

                // Create a moving color pattern
                float time = frameNumber * 0.05f;
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        uchar b = static_cast<uchar>(128 + 127 * sin(x * 0.1f + time));
                        uchar g = static_cast<uchar>(128 + 127 * sin(y * 0.1f + time));
                        uchar r = static_cast<uchar>(128 + 127 * sin((x+y) * 0.1f + time));
                        frame.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
                    }
                }

                // Add frame counter text
                std::string text = "Frame: " + std::to_string(frameNumber);
                cv::putText(frame, text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                           cv::Scalar(255, 255, 255), 2);
            } else {
                // Capture frame from camera
                if (!cap.read(frame)) {
                    std::cerr << "Error: Could not read frame from camera" << std::endl;
                    std::cerr << "Switching to test pattern" << std::endl;
                    usingTestPattern = true;
                    cap.release();
                    continue;
                }

                // Ensure frame is BGR format
                if (frame.channels() != 3) {
                    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
                }
            }

            // Calculate number of packets needed for this frame
            size_t frameSize = frame.total() * frame.elemSize();
            size_t maxDataPerPacket = MAX_PACKET_SIZE - sizeof(PacketHeader);
            uint32_t totalPackets = (frameSize + maxDataPerPacket - 1) / maxDataPerPacket;

            // Prepare frame header
            FrameHeader frameHeader;
            frameHeader.frameNumber = frameNumber;
            frameHeader.width = frame.cols;
            frameHeader.height = frame.rows;
            frameHeader.channels = frame.channels();
            frameHeader.totalSize = frameSize;
            frameHeader.totalPackets = totalPackets;

            // Send frame header packet
            std::vector<char> headerPacket(sizeof(FrameHeader));
            memcpy(headerPacket.data(), &frameHeader, sizeof(FrameHeader));

            ssize_t sent = sendto(serverSocket, headerPacket.data(), headerPacket.size(), 0,
                                 (struct sockaddr*)&clientAddr, clientAddrLen);

            if (sent < 0) {
                std::cerr << "Error sending header packet: " << strerror(errno) << std::endl;
                continue;
            }

            // Add a minimal delay after sending header to avoid overwhelming the network
            std::this_thread::sleep_for(std::chrono::microseconds(500));

            // Send data packets
            const char* frameData = reinterpret_cast<const char*>(frame.data);
            int packetsSent = 0;

            for (uint32_t i = 0; i < totalPackets; i++) {
                uint32_t offset = i * maxDataPerPacket;
                uint32_t dataSize = std::min(maxDataPerPacket, frameSize - offset);

                // Prepare data packet header
                PacketHeader packetHeader;
                packetHeader.frameNumber = frameNumber;
                packetHeader.packetNumber = i;
                packetHeader.totalPackets = totalPackets;
                packetHeader.dataSize = dataSize;

                // Create buffer for data packet
                std::vector<char> dataPacket(sizeof(PacketHeader) + dataSize);
                memcpy(dataPacket.data(), &packetHeader, sizeof(PacketHeader));
                memcpy(dataPacket.data() + sizeof(PacketHeader), frameData + offset, dataSize);

                // Send data packet
                sent = sendto(serverSocket, dataPacket.data(), dataPacket.size(), 0,
                             (struct sockaddr*)&clientAddr, clientAddrLen);

                if (sent < 0) {
                    std::cerr << "Error sending data packet " << i << ": " << strerror(errno) << std::endl;
                    continue;
                }

                packetsSent++;

                // Add a minimal delay every 200 packets to avoid overwhelming the network
                if (i % 200 == 0 && i > 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
            }

            // Print stats every 30 frames
            if (frameNumber % 30 == 0) {
                std::cout << "Sent frame " << frameNumber << " (" << packetsSent << "/" << totalPackets << " packets)" << std::endl;
            }

            frameNumber++;

            // FPS control
            auto frameEnd = std::chrono::steady_clock::now();
            auto frameDuration = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart);
            if (frameDuration < frameInterval) {
                std::this_thread::sleep_for(frameInterval - frameDuration);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    // Clean up
    if (cap.isOpened()) {
        cap.release();
    }
    close(serverSocket);
    std::cout << "Server stopped" << std::endl;

    return 0;
}
