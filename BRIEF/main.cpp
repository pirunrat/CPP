#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <utility>

// Generate 2n random offset points for BRIEF
std::vector<std::pair<int, int>> generate_brief_pattern(int patch_size, int n) {
    std::vector<std::pair<int, int>> coords;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(-patch_size / 2, patch_size / 2);

    for (int i = 0; i < 2 * n; ++i) {
        int x = dist(gen);
        int y = dist(gen);
        coords.emplace_back(y, x); // (dy, dx)
    }

    return coords;
}

// Structure to store BRIEF descriptors and the valid keypoints
struct BriefResult {
    std::vector<std::vector<uint8_t>> descriptors;
    std::vector<cv::KeyPoint> valid_keypoints;
};

// Check if a keypoint is safely within image bounds
bool is_within_bounds(const cv::KeyPoint& kp, int img_cols, int img_rows, int half_patch) {
    int x = static_cast<int>(kp.pt.x);
    int y = static_cast<int>(kp.pt.y);
    return x - half_patch >= 0 && x + half_patch < img_cols &&
           y - half_patch >= 0 && y + half_patch < img_rows;
}

// Compute BRIEF descriptor for a single keypoint
std::vector<uint8_t> compute_single_descriptor(
    const cv::Mat& image,
    int x, int y,
    const std::vector<std::pair<int, int>>& pattern_A,
    const std::vector<std::pair<int, int>>& pattern_B)
{
    std::vector<uint8_t> descriptor;

    for (size_t i = 0; i < pattern_A.size(); ++i) {
        int dy1 = pattern_A[i].first;
        int dx1 = pattern_A[i].second;
        int dy2 = pattern_B[i].first;
        int dx2 = pattern_B[i].second;

        uchar p1 = image.at<uchar>(y + dy1, x + dx1);
        uchar p2 = image.at<uchar>(y + dy2, x + dx2);

        descriptor.push_back(p1 < p2 ? 1 : 0);
    }

    return descriptor;
}

// Main function to compute BRIEF descriptors for a list of keypoints
BriefResult compute_brief_descriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<std::pair<int, int>>& pattern_A,
    const std::vector<std::pair<int, int>>& pattern_B,
    int patch_size = 31)
{
    std::vector<std::vector<uint8_t>> descriptors;
    std::vector<cv::KeyPoint> valid_keypoints;
    int half_patch = patch_size / 2;

    for (const auto& kp : keypoints) {
        if (!is_within_bounds(kp, image.cols, image.rows, half_patch))
            continue;

        int x = static_cast<int>(kp.pt.x);
        int y = static_cast<int>(kp.pt.y);

        std::vector<uint8_t> desc = compute_single_descriptor(image, x, y, pattern_A, pattern_B);

        descriptors.push_back(desc);
        valid_keypoints.push_back(kp);
    }

    return {descriptors, valid_keypoints};
}


int main() {
    // Load grayscale image
    cv::Mat image = cv::imread("../Test/download.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    // Detect keypoints using FAST
    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
    fast->detect(image, keypoints);

    // Generate BRIEF pattern
    int n = 256;
    auto coords = generate_brief_pattern(31, n);
    std::vector<std::pair<int, int>> pattern_A(coords.begin(), coords.begin() + n);
    std::vector<std::pair<int, int>> pattern_B(coords.begin() + n, coords.end());

    // Compute descriptors
    BriefResult result = compute_brief_descriptors(image, keypoints, pattern_A, pattern_B);

    // Draw keypoints
    cv::Mat output;
    cv::drawKeypoints(image, result.valid_keypoints, output, cv::Scalar(0, 255, 0));
    cv::imshow("BRIEF Keypoints", output);
    cv::waitKey(0);

    std::cout << "Computed descriptors: " << result.descriptors.size() << std::endl;

    return 0;
}
