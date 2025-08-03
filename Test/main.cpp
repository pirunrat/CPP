#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

int main() {
    // Open webcam
    std::cout << "🎥 Trying to open webcam...\n";
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "❌ Webcam could not be opened.\n";
        return -1;
    }
    std::cout << "✅ Webcam opened successfully.\n";

    // Create TCP socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "❌ Socket creation failed.\n";
        return -1;
    }

    // Set up server address
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(5001);  // Port must match Python
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

    // Connect to Python receiver
    std::cout << "🔌 Connecting to Python receiver...\n";
    if (connect(sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("❌ Connect failed");
        return -1;
    }
    std::cout << "✅ Connected to Python receiver.\n";

    while (true) {
        // Capture frame
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "⚠️ Empty frame captured.\n";
            continue;
        }

        // Resize for performance
        cv::resize(frame, frame, cv::Size(320, 240));

        // Encode as JPEG
        std::vector<uchar> buf;
        if (!cv::imencode(".jpg", frame, buf)) {
            std::cerr << "⚠️ Failed to encode frame.\n";
            continue;
        }

        // Send frame size in network byte order (big-endian)
        int size = buf.size();
        uint32_t be_size = htonl(size);  // 🔧 Big-endian conversion

        std::cout << "📤 Sending frame of size: " << size << " bytes\n";

        // Send frame size
        if (send(sock, &be_size, sizeof(be_size), 0) != sizeof(be_size)) {
            std::cerr << "❌ Failed to send frame size.\n";
            break;
        }

        // Send frame data
        if (send(sock, buf.data(), size, 0) != size) {
            std::cerr << "❌ Failed to send frame data.\n";
            break;
        }

        std::cout << "✅ Frame sent.\n";

        if (cv::waitKey(30) == 27) {  // ESC to stop
            std::cout << "🚪 ESC pressed. Exiting...\n";
            break;
        }
    }

    close(sock);
    return 0;
}
