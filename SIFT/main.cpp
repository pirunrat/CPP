#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace cv;
using namespace std;

// ---------- Utility ----------
float normL2(const vector<float>& v) {
    float sum = 0.0f;
    for (auto val : v) sum += val * val;
    return sqrt(sum);
}

// ---------- Gaussian Blur ----------
Mat applyGaussianBlur(const Mat& image, double sigma) {
    Mat result;
    int ksize = cvRound(sigma * 6) | 1;
    GaussianBlur(image, result, Size(ksize, ksize), sigma, sigma);
    return result;
}

// ---------- Scale Space ----------
vector<Mat> generateScaleSpace(const Mat& image, int num_scales = 5, double base_sigma = 1.6) {
    vector<Mat> scale_space;
    double k = pow(2.0, 1.0 / (num_scales - 3));
    for (int i = 0; i < num_scales; ++i) {
        double sigma = base_sigma * pow(k, i);
        scale_space.push_back(applyGaussianBlur(image, sigma));
    }
    return scale_space;
}

// ---------- DoG ----------
vector<Mat> computeDoG(const vector<Mat>& scale_space) {
    vector<Mat> dogs;
    for (size_t i = 1; i < scale_space.size(); ++i) {
        Mat diff;
        subtract(scale_space[i], scale_space[i - 1], diff);
        dogs.push_back(diff);
    }
    return dogs;
}

// ---------- Keypoints ----------
vector<Point> findKeypoints(const vector<Mat>& dogs, double threshold = 0.03) {
    vector<Point> keypoints;
    for (int i = 1; i < dogs.size() - 1; ++i) {
        for (int y = 1; y < dogs[i].rows - 1; ++y) {
            for (int x = 1; x < dogs[i].cols - 1; ++x) {
                float val = dogs[i].at<float>(y, x);
                if (fabs(val) < threshold) continue;

                bool is_extremum = true;
                for (int di = -1; di <= 1 && is_extremum; ++di)
                    for (int dy = -1; dy <= 1 && is_extremum; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (di == 0 && dx == 0 && dy == 0) continue;
                            float neighbor = dogs[i + di].at<float>(y + dy, x + dx);
                            if ((val > 0 && val < neighbor) || (val < 0 && val > neighbor)) {
                                is_extremum = false;
                                break;
                            }
                        }

                if (is_extremum) {
                    keypoints.emplace_back(x, y);
                }
            }
        }
    }
    return keypoints;
}

// ---------- Orientation ----------
float computeOrientation(const Mat& img, int x, int y, int radius = 8) {
    Mat gx, gy;
    Sobel(img, gx, CV_32F, 1, 0, 3);
    Sobel(img, gy, CV_32F, 0, 1, 3);

    vector<float> hist(36, 0);

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= img.cols || ny >= img.rows) continue;

            float mag = hypot(gx.at<float>(ny, nx), gy.at<float>(ny, nx));
            float angle = atan2(gy.at<float>(ny, nx), gx.at<float>(ny, nx)) * 180 / CV_PI;
            int bin = cvRound((angle + 360.0f) / 10.0f) % 36;
            hist[bin] += mag;
        }
    }

    int max_bin = max_element(hist.begin(), hist.end()) - hist.begin();
    return max_bin * 10.0f;
}




// ---------- Descriptor ----------
vector<float> extractDescriptor(const Mat& img, int x, int y, float orientation, int size = 16) {
    int half = size / 2;
    if (x - half < 0 || y - half < 0 || x + half >= img.cols || y + half >= img.rows)
        return {};

    Mat patch = img(Rect(x - half, y - half, size, size)).clone();
    Mat gx, gy;
    Sobel(patch, gx, CV_32F, 1, 0, 3);
    Sobel(patch, gy, CV_32F, 0, 1, 3);

    Mat mag, angle;
    cartToPolar(gx, gy, mag, angle, true);
    angle -= orientation;

    for (int i = 0; i < angle.rows; ++i) {
        for (int j = 0; j < angle.cols; ++j) {
            float& ang = angle.at<float>(i, j);
            ang = fmod(ang + 360.0f, 360.0f);
        }
    }

    vector<float> descriptor;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            vector<float> hist(8, 0);
            for (int u = 0; u < 4; ++u) {
                for (int v = 0; v < 4; ++v) {
                    int px = j * 4 + v;
                    int py = i * 4 + u;
                    float m = mag.at<float>(py, px);
                    float a = angle.at<float>(py, px);
                    int bin = static_cast<int>(floor(a / 45.0f)) % 8;
                    hist[bin] += m;
                }
            }
            descriptor.insert(descriptor.end(), hist.begin(), hist.end());
        }
    }

    float n = normL2(descriptor);
    if (n > 0) {
        for (auto& d : descriptor) d /= n;
        for (auto& d : descriptor) d = min(d, 0.2f);
        n = normL2(descriptor);
        for (auto& d : descriptor) d /= n;
    }

    return descriptor;
}




// // ---------- MAIN ----------
// int main() {
//     VideoCapture cap(0, cv::CAP_V4L2);
//     cap.set(CAP_PROP_FRAME_WIDTH, 640);
//     cap.set(CAP_PROP_FRAME_HEIGHT, 480);
//     cap.set(CAP_PROP_BUFFERSIZE, 1);

//     if (!cap.isOpened()) {
//         cerr << "Cannot open camera!" << endl;
//         return -1;
//     }

//     Mat frame, gray;
//     while (true) {
//         cap >> frame;
//         if (frame.empty()) break;

//         cvtColor(frame, gray, COLOR_BGR2GRAY);
//         resize(gray, gray, Size(320, 240));
//         gray.convertTo(gray, CV_32F, 1.0 / 255.0);

//         auto scale_space = generateScaleSpace(gray);
//         auto dogs = computeDoG(scale_space);
//         auto keypoints = findKeypoints(dogs);

//         Mat vis;
//         gray.convertTo(vis, CV_8U, 255.0);
//         cvtColor(vis, vis, COLOR_GRAY2BGR);

//         for (const auto& pt : keypoints) {
//             float angle = computeOrientation(scale_space[2], pt.x, pt.y);
//             auto desc = extractDescriptor(scale_space[2], pt.x, pt.y, angle);
//             if (!desc.empty()) {
//                 circle(vis, pt, 2, Scalar(0, 0, 255), -1);
//             }
//         }

//         imshow("SIFT Keypoints", vis);
//         if (waitKey(1) == 27) break; // ESC to exit
//     }

//     cap.release();
//     destroyAllWindows();
//     return 0;
// }

int main() {
    cv::Mat image = cv::imread("../Test/download.jpg");
    cv::Mat gray;

    if (image.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }

    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(320, 240));
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);

    auto scale_space = generateScaleSpace(gray);
    auto dogs = computeDoG(scale_space);
    auto keypoints = findKeypoints(dogs);

    cv::Mat vis;
    gray.convertTo(vis, CV_8U, 255.0);
    cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);

    for (const auto& pt : keypoints) {
    float angle = computeOrientation(scale_space[2], pt.x, pt.y);
    auto desc = extractDescriptor(scale_space[2], pt.x, pt.y, angle);

        if (!desc.empty()) {
            circle(vis, pt, 2, Scalar(0, 0, 255), -1);

            // Print descriptor
            cout << "Keypoint at (" << pt.x << ", " << pt.y << ") - Descriptor:" << endl;
            for (int i = 0; i < desc.size(); ++i) {
                cout << fixed << setprecision(3) << desc[i] << " ";
                if ((i + 1) % 16 == 0) cout << endl;  // 128 dims in 8x16 rows
            }
            cout << endl << "------------------------------" << endl;
        }
    }

    cv::imshow("SIFT Keypoints", vis);
    cv::waitKey(0);  // <-- keep window open
    return 0;
}

