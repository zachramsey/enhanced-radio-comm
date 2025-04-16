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

// Frame header structure (must match server)
struct FrameHeader {
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t dataSize;
};

// Function to receive all bytes
bool recvAll(int socket, char* buffer, size_t length) {
    size_t totalReceived = 0;
    while (totalReceived < length) {
        ssize_t received = recv(socket, buffer + totalReceived, length - totalReceived, 0);
        if (received <= 0) {
            if (received == 0) {
                std::cerr << "Connection closed by server" << std::endl;
            } else {
                std::cerr << "Error receiving data: " << strerror(errno) << std::endl;
            }
            return false;
        }
        totalReceived += received;
    }
    return true;
}

int main(int argc, char* argv[]) {
    // Default parameters
    std::string serverIP = "127.0.0.1";
    int port = 8888;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--ip" && i + 1 < argc) {
            serverIP = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --ip IP           Server IP address (default: 127.0.0.1)\n"
                      << "  --port PORT       Server port (default: 8888)\n"
                      << "  --help            Show this help message\n";
            return 0;
        }
    }
    
    // Create socket
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket < 0) {
        std::cerr << "Error creating socket: " << strerror(errno) << std::endl;
        return 1;
    }
    
    // Connect to server
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, serverIP.c_str(), &serverAddr.sin_addr) <= 0) {
        std::cerr << "Invalid address or address not supported" << std::endl;
        close(clientSocket);
        return 1;
    }
    
    std::cout << "Connecting to server at " << serverIP << ":" << port << "..." << std::endl;
    
    if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Connection failed: " << strerror(errno) << std::endl;
        close(clientSocket);
        return 1;
    }
    
    std::cout << "Connected to server" << std::endl;
    
    // Create window for display
    cv::namedWindow("Raw Video Stream", cv::WINDOW_NORMAL);
    
    // Variables for FPS calculation
    int frameCount = 0;
    auto fpsStartTime = std::chrono::steady_clock::now();
    double fps = 0.0;
    
    // Variables for data rate calculation
    size_t totalBytesReceived = 0;
    auto dataRateStartTime = std::chrono::steady_clock::now();
    double dataRateMbps = 0.0;
    
    // Streaming loop
    try {
        FrameHeader header;
        std::vector<char> frameBuffer;
        
        while (true) {
            // Receive header
            if (!recvAll(clientSocket, reinterpret_cast<char*>(&header), sizeof(header))) {
                break;
            }
            
            // Validate header
            if (header.width <= 0 || header.height <= 0 || header.channels <= 0 || header.dataSize <= 0) {
                std::cerr << "Invalid header received" << std::endl;
                break;
            }
            
            // Resize buffer if needed
            if (frameBuffer.size() < header.dataSize) {
                frameBuffer.resize(header.dataSize);
            }
            
            // Receive frame data
            if (!recvAll(clientSocket, frameBuffer.data(), header.dataSize)) {
                break;
            }
            
            // Update data rate statistics
            totalBytesReceived += sizeof(header) + header.dataSize;
            auto now = std::chrono::steady_clock::now();
            auto dataRateElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - dataRateStartTime).count();
            if (dataRateElapsed >= 1) {
                dataRateMbps = (totalBytesReceived * 8.0 / 1000000.0) / dataRateElapsed;
                totalBytesReceived = 0;
                dataRateStartTime = now;
            }
            
            // Create Mat from received data
            cv::Mat frame(header.height, header.width, CV_8UC3, frameBuffer.data());
            
            // Update FPS counter
            frameCount++;
            auto fpsElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - fpsStartTime).count();
            if (fpsElapsed >= 1) {
                fps = frameCount / static_cast<double>(fpsElapsed);
                frameCount = 0;
                fpsStartTime = now;
            }
            
            // Display stats on frame
            std::string statsText = cv::format("Resolution: %dx%d | FPS: %.1f | Data Rate: %.2f Mbps", 
                                              header.width, header.height, fps, dataRateMbps);
            cv::putText(frame, statsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                       cv::Scalar(0, 255, 0), 2);
            
            // Display frame
            cv::imshow("Raw Video Stream", frame);
            
            // Check for key press (ESC to exit)
            int key = cv::waitKey(1);
            if (key == 27) {  // ESC key
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    
    // Clean up
    close(clientSocket);
    cv::destroyAllWindows();
    std::cout << "Disconnected from server" << std::endl;
    
    return 0;
}
