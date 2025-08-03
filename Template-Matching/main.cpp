#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// Global state
cv::Rect selectedROI;
bool drawing = false;
bool templateSelected = false;
cv::Mat templateROI;
cv::Mat processedTemplate;

// Mouse callback to draw ROI
void mouseHandler(int event, int x, int y, int flags, void* userdata) {
    static cv::Point origin;
    if (event == cv::EVENT_LBUTTONDOWN) {
        drawing = true;
        origin = cv::Point(x, y);
        selectedROI = cv::Rect(x, y, 0, 0);
    } else if (event == cv::EVENT_MOUSEMOVE && drawing) {
        selectedROI = cv::Rect(origin, cv::Point(x, y));
    } else if (event == cv::EVENT_LBUTTONUP) {
        drawing = false;
        selectedROI = cv::Rect(origin, cv::Point(x, y));
        templateSelected = true;
    }
}

// Flatten and normalize a patch to a 1D vector
std::vector<float> flattenAndNormalize(const cv::Mat& patch) {
    cv::Mat continuousPatch = patch.isContinuous() ? patch : patch.clone();

    std::vector<float> vec;
    vec.reserve(continuousPatch.total());
    float norm = 0.0f;

    for (int i = 0; i < continuousPatch.total(); ++i) {
        float val = static_cast<float>(continuousPatch.ptr<uchar>()[i]);
        vec.push_back(val);
        norm += val * val;
    }

    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& v : vec) v /= norm;
    }

    return vec;
}

// Cosine similarity between two vectors
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        dot += a[i] * b[i];
    return dot;
}

// Template matching using cosine similarity with downscaled frame
void match(cv::Mat& frame, const cv::Mat& templ, float threshold = 0.7f) {
    CV_Assert(!templ.empty() && templ.type() == CV_8UC1);

    float scale = 0.5f;
    cv::Mat smallFrame, graySmall;
    cv::resize(frame, smallFrame, cv::Size(), scale, scale);
    cv::cvtColor(smallFrame, graySmall, cv::COLOR_BGR2GRAY);

    cv::Mat templSmall;
    cv::resize(templ, templSmall, cv::Size(), scale, scale);

    int th = templSmall.rows;
    int tw = templSmall.cols;
    std::vector<float> templVec = flattenAndNormalize(templSmall);

    float bestScore = -1.0f;
    cv::Point bestPoint;

    int step_size = 16;
    for (int i = 0; i <= graySmall.rows - th; i += step_size) {
        for (int j = 0; j <= graySmall.cols - tw; j += step_size) {
            cv::Mat patch = graySmall(cv::Rect(j, i, tw, th));
            std::vector<float> patchVec = flattenAndNormalize(patch);
            float score = cosineSimilarity(patchVec, templVec);

            if (score > bestScore) {
                bestScore = score;
                bestPoint = cv::Point(j, i);
            }
        }
    }

    if (bestScore >= threshold) {
        cv::Point topLeft(bestPoint.x / scale, bestPoint.y / scale);
        cv::Point bottomRight((bestPoint.x + tw) / scale, (bestPoint.y + th) / scale);

        cv::rectangle(frame, topLeft, bottomRight, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, cv::format("Score: %.2f", bestScore),
                    topLeft + cv::Point(0, -10), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(255, 255, 255), 2);
    } else {
        cv::putText(frame, cv::format("Low match: %.2f", bestScore), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }
}

int main() {
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open webcam.\n";
        return -1;
    }

    cv::namedWindow("Live");
    cv::setMouseCallback("Live", mouseHandler);

    int frameCount = 0;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        frameCount++;

        if (drawing) {
            cv::Mat temp = frame.clone();
            cv::rectangle(temp, selectedROI, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Live", temp);
        } else {
            if (templateSelected) {
                cv::Mat gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                templateROI = gray(selectedROI).clone();
                processedTemplate = templateROI.clone();  // Pre-process once
                templateSelected = false;
                std::cout << "[INFO] Template selected, matching started.\n";
            }

            if (!processedTemplate.empty() && frameCount % 1 == 0) {
                match(frame, processedTemplate, 0.8f);
            }

            cv::imshow("Live", frame);
        }

        if (cv::waitKey(1) == 27) break; // ESC to quit
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
