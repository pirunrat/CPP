#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

Mat toGrayscale(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    return gray;
}

Mat applyGaussianBlur(const Mat& img) {
    Mat blurred;
    GaussianBlur(img, blurred, Size(5, 5), 1.4);
    return blurred;
}

void computeGradients(const Mat& img, Mat& magnitude, Mat& angle) {
    Mat gradX, gradY;
    Sobel(img, gradX, CV_64F, 1, 0);
    Sobel(img, gradY, CV_64F, 0, 1);

    magnitude = Mat::zeros(img.size(), CV_64F);
    angle = Mat::zeros(img.size(), CV_64F);

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            double gx = gradX.at<double>(i, j);
            double gy = gradY.at<double>(i, j);
            magnitude.at<double>(i, j) = hypot(gx, gy);
            angle.at<double>(i, j) = atan2(gy, gx) * 180.0 / CV_PI;
            if (angle.at<double>(i, j) < 0)
                angle.at<double>(i, j) += 180;
        }
    }
}

Mat nonMaximumSuppression(const Mat& mag, const Mat& angle) {
    Mat suppressed = Mat::zeros(mag.size(), CV_64F);
    for (int i = 1; i < mag.rows - 1; ++i) {
        for (int j = 1; j < mag.cols - 1; ++j) {
            double a = angle.at<double>(i, j);
            double m = mag.at<double>(i, j);
            double q = 255, r = 255;

            if ((0 <= a && a < 22.5) || (157.5 <= a && a <= 180)) {
                q = mag.at<double>(i, j + 1);
                r = mag.at<double>(i, j - 1);
            } else if (22.5 <= a && a < 67.5) {
                q = mag.at<double>(i + 1, j - 1);
                r = mag.at<double>(i - 1, j + 1);
            } else if (67.5 <= a && a < 112.5) {
                q = mag.at<double>(i + 1, j);
                r = mag.at<double>(i - 1, j);
            } else if (112.5 <= a && a < 157.5) {
                q = mag.at<double>(i - 1, j - 1);
                r = mag.at<double>(i + 1, j + 1);
            }

            if (m >= q && m >= r)
                suppressed.at<double>(i, j) = m;
            else
                suppressed.at<double>(i, j) = 0;
        }
    }
    return suppressed;
}

Mat hysteresis(const Mat& img, double lowThresh, double highThresh) {
    Mat result = Mat::zeros(img.size(), CV_8U);
    const uchar WEAK = 50;
    const uchar STRONG = 255;

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            double val = img.at<double>(i, j);
            if (val >= highThresh) {
                result.at<uchar>(i, j) = STRONG;
            } else if (val >= lowThresh) {
                result.at<uchar>(i, j) = WEAK;
            }
        }
    }

    for (int i = 1; i < img.rows - 1; ++i) {
        for (int j = 1; j < img.cols - 1; ++j) {
            if (result.at<uchar>(i, j) == WEAK) {
                bool connected = false;
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (result.at<uchar>(i + di, j + dj) == STRONG) {
                            connected = true;
                            break;
                        }
                    }
                }
                result.at<uchar>(i, j) = connected ? STRONG : 0;
            }
        }
    }

    return result;
}

Mat cannyFromScratch(const Mat& input, double lowThresh, double highThresh) {
    Mat gray = toGrayscale(input);
    Mat blurred = applyGaussianBlur(gray);

    Mat mag, angle;
    computeGradients(blurred, mag, angle);

    Mat suppressed = nonMaximumSuppression(mag, angle);
    Mat edges = hysteresis(suppressed, lowThresh, highThresh);

    return edges;
}

int main() {
    VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_BUFFERSIZE, 1);
    if (!cap.isOpened()) {
        cerr << "Cannot open webcam!" << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        Mat edges = cannyFromScratch(frame, 50, 150);

        imshow("Canny Edge Detection (from Scratch)", edges);
        if (waitKey(1) == 27) break;  // ESC to quit
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
