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

// Frame header structure
struct FrameHeader {
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t dataSize;
};

// Function to send all bytes
bool sendAll(int socket, const char* buffer, size_t length) {
    size_t totalSent = 0;
    while (totalSent < length) {
        ssize_t sent = send(socket, buffer + totalSent, length - totalSent, 0);
        if (sent < 0) {
            std::cerr << "Error sending data: " << strerror(errno) << std::endl;
            return false;
        }
        totalSent += sent;
    }
    return true;
}

int main(int argc, char* argv[]) {
    // Default parameters
    int port = 8888;
    int cameraIndex = 0;
    int width = 640;
    int height = 480;
    int fps = 30;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--camera" && i + 1 < argc) {
            cameraIndex = std::stoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            fps = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --port PORT       Port to listen on (default: 8888)\n"
                      << "  --camera INDEX    Camera device index (default: 0)\n"
                      << "  --width WIDTH     Frame width (default: 640)\n"
                      << "  --height HEIGHT   Frame height (default: 480)\n"
                      << "  --fps FPS         Target frames per second (default: 30)\n"
                      << "  --help            Show this help message\n";
            return 0;
        }
    }
    
    // Create socket
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0) {
        std::cerr << "Error creating socket: " << strerror(errno) << std::endl;
        return 1;
    }
    
    // Set socket options to allow reuse of address
    int opt = 1;
    if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "Error setting socket options: " << strerror(errno) << std::endl;
        close(serverSocket);
        return 1;
    }
    
    // Bind socket to port
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(port);
    
    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Error binding socket: " << strerror(errno) << std::endl;
        close(serverSocket);
        return 1;
    }
    
    // Listen for connections
    if (listen(serverSocket, 5) < 0) {
        std::cerr << "Error listening on socket: " << strerror(errno) << std::endl;
        close(serverSocket);
        return 1;
    }
    
    std::cout << "Server started. Listening on port " << port << std::endl;
    std::cout << "Camera: " << cameraIndex << ", Resolution: " << width << "x" << height << ", FPS: " << fps << std::endl;
    
    // Main server loop
    while (true) {
        std::cout << "Waiting for client connection..." << std::endl;
        
        // Accept client connection
        struct sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);
        
        if (clientSocket < 0) {
            std::cerr << "Error accepting connection: " << strerror(errno) << std::endl;
            continue;
        }
        
        std::cout << "Client connected: " << inet_ntoa(clientAddr.sin_addr) << std::endl;
        
        // Open camera
        cv::VideoCapture cap(cameraIndex);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera " << cameraIndex << std::endl;
            send(clientSocket, "ERROR: Could not open camera", 28, 0);
            close(clientSocket);
            continue;
        }
        
        // Set camera properties
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap.set(cv::CAP_PROP_FPS, fps);
        
        // Get actual camera properties (may differ from requested)
        width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        
        std::cout << "Camera opened. Actual resolution: " << width << "x" << height << std::endl;
        
        // Calculate frame interval for FPS control
        std::chrono::microseconds frameInterval(1000000 / fps);
        
        // Streaming loop
        std::atomic<bool> clientConnected{true};
        
        // Start a thread to check if client is still connected
        std::thread connectionChecker([&clientConnected, clientSocket]() {
            char buffer[1];
            while (clientConnected) {
                // Try to peek at the socket
                int result = recv(clientSocket, buffer, 1, MSG_PEEK | MSG_DONTWAIT);
                if (result == 0 || (result < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                    // Client disconnected
                    clientConnected = false;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
        
        try {
            cv::Mat frame;
            FrameHeader header;
            
            while (clientConnected) {
                auto frameStart = std::chrono::steady_clock::now();
                
                // Capture frame
                if (!cap.read(frame)) {
                    std::cerr << "Error: Could not read frame from camera" << std::endl;
                    break;
                }
                
                // Ensure frame is BGR format
                if (frame.channels() != 3) {
                    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
                }
                
                // Prepare header
                header.width = frame.cols;
                header.height = frame.rows;
                header.channels = frame.channels();
                header.dataSize = frame.total() * frame.elemSize();
                
                // Send header
                if (!sendAll(clientSocket, reinterpret_cast<const char*>(&header), sizeof(header))) {
                    std::cerr << "Error sending header" << std::endl;
                    break;
                }
                
                // Send frame data
                if (!sendAll(clientSocket, reinterpret_cast<const char*>(frame.data), header.dataSize)) {
                    std::cerr << "Error sending frame data" << std::endl;
                    break;
                }
                
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
        clientConnected = false;
        connectionChecker.join();
        cap.release();
        close(clientSocket);
        std::cout << "Client disconnected" << std::endl;
    }
    
    // Close server socket
    close(serverSocket);
    return 0;
}
