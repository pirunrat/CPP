#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

// Binary threshold function that converts to grayscale if needed
void BinaryThreshold(Mat &image) {
    // Convert to grayscale if not already
    if (image.channels() != 1) {
        cvtColor(image, image, COLOR_BGR2GRAY);
    }

    // Apply manual binary threshold
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            uchar &pixel = image.at<uchar>(i, j);
            pixel = (pixel > 128) ? 255 : 0;
        }
    }
}

int main() {
    // Open webcam using V4L2 backend
    VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_BUFFERSIZE, 1);

    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera\n";
        return -1;
    }

    Mat frame;
    while (true) {
        cap.read(frame);
        if (frame.empty()) break;

        BinaryThreshold(frame);  // Convert to binary threshold

        imshow("Binary Webcam", frame);  // Show thresholded image

        if (waitKey(1) == 27) break;  // ESC to exit
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
