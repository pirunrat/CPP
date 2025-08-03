#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>
#include <cmath>
#include <map>
#include <vector>

// Convert to grayscale
cv::Mat convertToGray(const cv::Mat &image) {
    CV_Assert(image.type() == CV_8UC3);
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

// Threshold to binary
cv::Mat convertBinary(cv::Mat &gray) {
    CV_Assert(gray.type() == CV_8UC1);
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    return binary;
}

// Morphological opening (remove noise)
cv::Mat morphologicalOpening(cv::Mat &binary) {
    cv::Mat processedImage;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, processedImage, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);
    return processedImage;
}

// Dilation to get sure background
cv::Mat dilation(cv::Mat &binary) {
    cv::Mat sure_bg;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binary, sure_bg, kernel, cv::Point(-1, -1), 3);
    return sure_bg;
}

// Breadth-First Search for distance transform
float BFS(const cv::Mat& binary, int start_i, int start_j) {
    int row = binary.rows;
    int col = binary.cols;
    std::queue<std::pair<cv::Point, int>> q;
    q.push({cv::Point(start_j, start_i), 0});
    cv::Mat visited = cv::Mat::zeros(row, col, CV_8U);

    while (!q.empty()) {
        auto [pt, dist] = q.front(); q.pop();
        int x = pt.x, y = pt.y;
        if (visited.at<uchar>(y, x)) continue;
        visited.at<uchar>(y, x) = 1;

        if (binary.at<uchar>(y, x) == 0) return static_cast<float>(dist);

        std::vector<cv::Point> directions = {{1,0},{-1,0},{0,1},{0,-1}};
        for (auto dir : directions) {
            int nx = x + dir.x;
            int ny = y + dir.y;
            if (nx >= 0 && nx < col && ny >= 0 && ny < row && !visited.at<uchar>(ny, nx)) {
                q.push({cv::Point(nx, ny), dist + 1});
            }
        }
    }
    return -1.0f;
}

// Custom distance transform
cv::Mat distanceTransform(cv::Mat binary) {
    int row = binary.rows;
    int col = binary.cols;
    cv::Mat output = cv::Mat::zeros(binary.size(), CV_32FC1);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (binary.at<uchar>(i, j) != 0) {
                output.at<float>(i, j) = BFS(binary, i, j);
            }
        }
    }

    // Normalize distance map
    cv::normalize(output, output, 0, 1.0, cv::NORM_MINMAX);
    return output;
}

// Simplified watershed implementation
void applyWatershedFromScratch(cv::Mat& image, const cv::Mat& sure_fg, const cv::Mat& sure_bg) {
    CV_Assert(sure_fg.type() == CV_8U && sure_bg.type() == CV_8U);

    // Step 1: Unknown area = sure_bg - sure_fg
    cv::Mat unknown;
    cv::subtract(sure_bg, sure_fg, unknown);

    // Step 2: Label sure foreground
    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers);
    markers += 1; // make background = 1, others ≥ 2

    // Set unknown region to 0
    for (int i = 0; i < unknown.rows; ++i) {
        for (int j = 0; j < unknown.cols; ++j) {
            if (unknown.at<uchar>(i, j) == 255) {
                markers.at<int>(i, j) = 0;
            }
        }
    }

    // Step 3: Region growing from markers
    cv::Mat result = markers.clone();
    std::queue<cv::Point> q;

    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            if (result.at<int>(i, j) > 1) {
                q.push(cv::Point(j, i));
            }
        }
    }

    std::vector<cv::Point> directions = {
        {1,0}, {-1,0}, {0,1}, {0,-1}
    };

    while (!q.empty()) {
        cv::Point p = q.front(); q.pop();
        int currentLabel = result.at<int>(p.y, p.x);

        for (const auto& d : directions) {
            int nx = p.x + d.x;
            int ny = p.y + d.y;

            if (nx >= 0 && ny >= 0 && nx < result.cols && ny < result.rows) {
                if (result.at<int>(ny, nx) == 0) {
                    result.at<int>(ny, nx) = currentLabel;
                    q.push(cv::Point(nx, ny));
                }
            }
        }
    }

    // Step 4: Detect boundaries between different labels
    for (int i = 1; i < result.rows - 1; ++i) {
        for (int j = 1; j < result.cols - 1; ++j) {
            int current = result.at<int>(i, j);
            for (const auto& d : directions) {
                int neighbor = result.at<int>(i + d.y, j + d.x);
                if (neighbor != current && neighbor > 0 && current > 0) {
                    image.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255); // Mark boundary red
                    break;
                }
            }
        }
    }
}


int main() {
    cv::Mat image = cv::imread("../../Test/download.jpg");
    if (image.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }

    // Step-by-step pipeline
    cv::Mat gray = convertToGray(image);
    cv::Mat binary = convertBinary(gray);
    cv::Mat opened = morphologicalOpening(binary);
    cv::Mat sure_bg = dilation(opened);
    cv::Mat dist = distanceTransform(opened);

    // Convert distance map to 8-bit for foreground extraction
    cv::Mat sure_fg;
    cv::threshold(dist, sure_fg, 0.5, 1.0, cv::THRESH_BINARY);
    sure_fg.convertTo(sure_fg, CV_8U);

    applyWatershedFromScratch(image, sure_fg, sure_bg);

    cv::imshow("Watershed Segmentation", image);
    cv::waitKey(0);
    return 0;
}
