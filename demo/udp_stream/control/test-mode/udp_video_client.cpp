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
#include <cmath>

// Default settings
constexpr int DEFAULT_PORT         = 8888;
constexpr int DEFAULT_BUFFER_SIZE  = 10;
constexpr int DEFAULT_WIDTH        = 640;
constexpr int DEFAULT_HEIGHT       = 480;

// Maximum UDP packet size (slightly less than typical MTU to avoid fragmentation)
constexpr size_t MAX_PACKET_SIZE = 1400;

// Frame header structure (must match server)
struct FrameHeader {
    uint32_t frameNumber;
    int32_t  width;
    int32_t  height;
    int32_t  channels;
    int32_t  totalSize;
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
    FrameBuffer(int maxSize)
      : maxBufferSize(maxSize), running(true) {}

    void addPacket(const PacketHeader& header, const char* data, size_t dataSize) {
        std::unique_lock<std::mutex> lock(mutex);
        auto& frame = frames[header.frameNumber];
        if (frame.totalPackets == 0) {
            frame.totalPackets   = header.totalPackets;
            frame.receivedPackets= 0;
            frame.data.resize(header.totalPackets);
            frame.received.assign(header.totalPackets, false);
        }
        if (!frame.received[header.packetNumber]) {
            frame.data[header.packetNumber].assign(data, data + dataSize);
            frame.received[header.packetNumber] = true;
            frame.receivedPackets++;
            if (frame.receivedPackets == frame.totalPackets) {
                completeFrames.push(header.frameNumber);
                condition.notify_one();
            }
        }
        cleanupOldFrames();
    }

    bool getNextFrame(cv::Mat& outFrame) {
        std::unique_lock<std::mutex> lock(mutex);
        while (completeFrames.empty() && running) {
            condition.wait_for(lock, std::chrono::milliseconds(100));
        }
        if (!running) return false;
        uint32_t fn = completeFrames.front();
        completeFrames.pop();
        auto frameData = std::move(frames[fn]);
        frames.erase(fn);

        size_t totalSize = 0;
        for (auto& pkt : frameData.data) totalSize += pkt.size();
        std::vector<char> buffer(totalSize);
        size_t offset = 0;
        for (auto& pkt : frameData.data) {
            memcpy(buffer.data() + offset, pkt.data(), pkt.size());
            offset += pkt.size();
        }

        if (frameData.width <= 0 || frameData.height <= 0 || frameData.channels != 3)
            return false;

        cv::Mat mat(frameData.height,
                    frameData.width,
                    CV_8UC3,
                    buffer.data());
        mat.copyTo(outFrame);
        return true;
    }

    void setFrameDimensions(uint32_t frameNumber, int w, int h, int c) {
        std::unique_lock<std::mutex> lock(mutex);
        auto& f = frames[frameNumber];
        f.width    = w;
        f.height   = h;
        f.channels = c;
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mutex);
        running = false;
        condition.notify_all();
    }

private:
    struct Frame {
        int width = 0, height = 0, channels = 0;
        uint32_t totalPackets = 0, receivedPackets = 0;
        std::vector<std::vector<char>> data;
        std::vector<bool> received;
    };

    void cleanupOldFrames() {
        if (frames.size() <= (size_t)maxBufferSize) return;
        auto it = frames.begin();
        while (frames.size() > (size_t)maxBufferSize && it != frames.end()) {
            it = frames.erase(it);
        }
    }

    std::mutex mutex;
    std::condition_variable condition;
    std::map<uint32_t, Frame> frames;
    std::queue<uint32_t>      completeFrames;
    int                       maxBufferSize;
    std::atomic<bool>         running;
};

// Helper to generate the same test‐pattern as server
static cv::Mat makeTestPattern(int width, int height, uint32_t frameNumber) {
    cv::Mat frame(height, width, CV_8UC3);
    float t = frameNumber * 0.05f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uchar b = static_cast<uchar>(128 + 127 * std::sin(x * 0.1f + t));
            uchar g = static_cast<uchar>(128 + 127 * std::sin(y * 0.1f + t));
            uchar r = static_cast<uchar>(128 + 127 * std::sin((x + y) * 0.1f + t));
            frame.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
        }
    }
    std::string txt = "Frame: " + std::to_string(frameNumber);
    cv::putText(frame, txt, cv::Point(20,30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(255,255,255), 2);
    return frame;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [--port PORT] --server IP [--buffer SIZE]\n";
}

int main(int argc, char* argv[]) {
    int port        = DEFAULT_PORT;
    std::string serverIP;
    int bufferSize  = DEFAULT_BUFFER_SIZE;

    const struct option opts[] = {
        {"port",   required_argument, nullptr, 'p'},
        {"server", required_argument, nullptr, 's'},
        {"buffer", required_argument, nullptr, 'b'},
        {"help",   no_argument,       nullptr, '?'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "p:s:b:?", opts, nullptr)) != -1) {
        switch (opt) {
            case 'p': port       = std::stoi(optarg); break;
            case 's': serverIP   = optarg;            break;
            case 'b': bufferSize = std::stoi(optarg); break;
            case '?': printUsage(argv[0]); return 0;
            default:  printUsage(argv[0]); return 1;
        }
    }

    if (serverIP.empty()) {
        std::cerr << "Error: --server IP is required\n";
        printUsage(argv[0]);
        return 1;
    }

    // Test‐pattern mode if IP == 0.0.0.0
    bool usingTestPattern = (serverIP == "0.0.0.0");
    if (usingTestPattern) {
        std::cout << "Test‐pattern mode enabled (no network)\n";
    }

    int clientSocket = -1;
    FrameBuffer* frameBuffer = nullptr;
    std::thread receiverThread;
    struct sockaddr_in serverAddr{};

    if (!usingTestPattern) {
        // Create socket
        clientSocket = socket(AF_INET, SOCK_DGRAM, 0);
        if (clientSocket < 0) {
            perror("socket");
            return 1;
        }
        // Allow reuse
        int yes = 1;
        setsockopt(clientSocket, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        // Bind
        struct sockaddr_in local{AF_INET, htons(port), INADDR_ANY};
        if (bind(clientSocket, (sockaddr*)&local, sizeof(local)) < 0) {
            perror("bind");
            close(clientSocket);
            return 1;
        }
        // Setup server address
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port   = htons(port);
        if (inet_pton(AF_INET, serverIP.c_str(), &serverAddr.sin_addr) <= 0) {
            std::cerr << "Invalid server IP\n";
            close(clientSocket);
            return 1;
        }
        // Send INIT
        const char* initMsg = "INIT";
        sendto(clientSocket, initMsg, strlen(initMsg), 0,
               (sockaddr*)&serverAddr, sizeof(serverAddr));

        // FrameBuffer + receiver
        frameBuffer = new FrameBuffer(bufferSize);
        receiverThread = std::thread([&]() {
            char buf[MAX_PACKET_SIZE];
            sockaddr_in src{};
            socklen_t slen = sizeof(src);
            while (true) {
                ssize_t len = recvfrom(clientSocket, buf, sizeof(buf), 0,
                                       (sockaddr*)&src, &slen);
                if (len < 0) break;
                if (len == sizeof(FrameHeader)) {
                    auto* h = reinterpret_cast<FrameHeader*>(buf);
                    frameBuffer->setFrameDimensions(
                        h->frameNumber, h->width, h->height, h->channels
                    );
                } else if (len > (ssize_t)sizeof(PacketHeader)) {
                    auto* p = reinterpret_cast<PacketHeader*>(buf);
                    frameBuffer->addPacket(
                        *p,
                        buf + sizeof(PacketHeader),
                        len - sizeof(PacketHeader)
                    );
                }
            }
        });
    }

    // Create display window
    cv::namedWindow("UDP Video Stream", cv::WINDOW_NORMAL);

    // Main display loop
    uint32_t frameNumber = 0;
    cv::Mat frame;
    while (true) {
        if (usingTestPattern) {
            frame = makeTestPattern(DEFAULT_WIDTH, DEFAULT_HEIGHT, frameNumber++);
        } else {
            if (!frameBuffer->getNextFrame(frame))
                break;
        }

        cv::imshow("UDP Video Stream", frame);
        if (cv::waitKey(1) == 27)  // ESC
            break;
    }

    // Cleanup
    if (!usingTestPattern) {
        frameBuffer->stop();
        receiverThread.join();
        close(clientSocket);
        delete frameBuffer;
    }
    cv::destroyAllWindows();
    return 0;
}
