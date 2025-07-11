#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>



cv::Mat X_gradient(const cv::Mat& image) {
    CV_Assert(image.type() == CV_8UC1);  // Ensure it's a grayscale image

    cv::Mat result = cv::Mat::zeros(image.size(), CV_32F);  // Store float gradient

    std::vector<int> kernel = {-1, 0, 1};

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 1; j < image.cols - 1; ++j) {  // avoid borders
            float val = 0;
            val += kernel[0] * image.at<uchar>(i, j - 1);
            val += kernel[1] * image.at<uchar>(i, j);
            val += kernel[2] * image.at<uchar>(i, j + 1);
            result.at<float>(i, j) = val;
        }
    }

    return result;
}



cv::Mat Y_gradient(const cv::Mat& image) {
    CV_Assert(image.type() == CV_8UC1);  // Ensure it's a grayscale image

    cv::Mat result = cv::Mat::zeros(image.size(), CV_32F);  // Store float gradient

    std::vector<int> kernel = {-1, 0, 1};

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 1; j < image.cols - 1; ++j) {  // avoid borders
            float val = 0;
            val += kernel[0] * image.at<uchar>(i-1, j);
            val += kernel[1] * image.at<uchar>(i, j);
            val += kernel[2] * image.at<uchar>(i+1, j);
            result.at<float>(i, j) = val;
        }
    }

    return result;
}



void computeStructureTensor(const cv::Mat& gray, cv::Mat& Ix2, cv::Mat& Iy2, cv::Mat& IxIy) {
    cv::Mat Ix = X_gradient(gray);
    cv::Mat Iy = Y_gradient(gray);

    cv::multiply(Ix, Ix, Ix2);
    cv::multiply(Iy, Iy, Iy2);
    cv::multiply(Ix, Iy, IxIy);

    // Smooth them
    cv::GaussianBlur(Ix2, Ix2, cv::Size(3, 3), 1);
    cv::GaussianBlur(Iy2, Iy2, cv::Size(3, 3), 1);
    cv::GaussianBlur(IxIy, IxIy, cv::Size(3, 3), 1);
}




cv::Mat computeHarrisResponse(const cv::Mat& Ix2, const cv::Mat& Iy2, const cv::Mat& IxIy, float k = 0.04f) {
    cv::Mat response = cv::Mat::zeros(Ix2.size(), CV_32F);

    for (int y = 0; y < Ix2.rows; ++y) {
        for (int x = 0; x < Ix2.cols; ++x) {
            float a = Ix2.at<float>(y, x);
            float b = IxIy.at<float>(y, x);
            float c = Iy2.at<float>(y, x);

            float det = a * c - b * b;
            float trace = a + c;

            response.at<float>(y, x) = det - k * trace * trace;
        }
    }

    return response;
}




cv::Mat drawHarrisCorners(const cv::Mat& gray, const cv::Mat& harris_response, float threshold_ratio = 0.01f) {
    cv::Mat output;
    cv::cvtColor(gray, output, cv::COLOR_GRAY2BGR);

    double max_response;
    cv::minMaxLoc(harris_response, nullptr, &max_response);

    for (int y = 0; y < harris_response.rows; ++y) {
        for (int x = 0; x < harris_response.cols; ++x) {
            if (harris_response.at<float>(y, x) > threshold_ratio * max_response) {
                cv::circle(output, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
            }
        }
    }

    return output;
}




int main() {
    cv::Mat img = cv::imread("../Test/download.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(320, 240));

    cv::Mat Ix2, Iy2, IxIy;
    computeStructureTensor(gray, Ix2, Iy2, IxIy);

    cv::Mat harris_response = computeHarrisResponse(Ix2, Iy2, IxIy);
    cv::Mat result = drawHarrisCorners(gray, harris_response);

    cv::imshow("Harris Corners", result);
    cv::waitKey(0);
    return 0;
}

