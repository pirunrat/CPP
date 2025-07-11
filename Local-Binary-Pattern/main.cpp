#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>


uchar rotateRight(uchar val, int n) {
    return (val >> n) | (val << (8 - n));
}


uchar rotationInvariantLBP(uchar code) {
    uchar minVal = code;
    for (int i = 1; i < 8; ++i) {
        uchar rotated = rotateRight(code, i);
        if (rotated < minVal)
            minVal = rotated;
    }
    return minVal;
}


int binaryToDecimal(const std::vector<int>& bits) {
    int result = 0;
    int n = bits.size();

    for (int i = 0; i < n; ++i) {
        result += bits[i] * std::pow(2, n - 1 - i);
    }

    return result;
}


cv::Mat Local_Binary_Pattern_RI(cv::Mat &gray) {
    CV_Assert(gray.type() == CV_8UC1);

    cv::Mat result = cv::Mat::zeros(gray.size(), CV_8UC1);
    int offset = 1;

    for (int i = offset; i < gray.rows - offset; ++i) {
        for (int j = offset; j < gray.cols - offset; ++j) {
            uchar center = gray.at<uchar>(i, j);
            uchar code = 0;

            code |= (gray.at<uchar>(i - 1, j - 1) >= center) << 7;
            code |= (gray.at<uchar>(i - 1, j)     >= center) << 6;
            code |= (gray.at<uchar>(i - 1, j + 1) >= center) << 5;
            code |= (gray.at<uchar>(i,     j + 1) >= center) << 4;
            code |= (gray.at<uchar>(i + 1, j + 1) >= center) << 3;
            code |= (gray.at<uchar>(i + 1, j)     >= center) << 2;
            code |= (gray.at<uchar>(i + 1, j - 1) >= center) << 1;
            code |= (gray.at<uchar>(i,     j - 1) >= center) << 0;

            result.at<uchar>(i, j) = rotationInvariantLBP(code);
        }
    }

    return result;
}




int main() {
    cv::Mat img = cv::imread("../Test/download.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }

    cv::Mat lbp = Local_Binary_Pattern_RI(img);
    cv::imshow("LBP Image", lbp);
    cv::waitKey(0);
    return 0;
}
