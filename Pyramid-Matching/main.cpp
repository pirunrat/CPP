#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <numeric>

// Global variables for mouse callback to select the template
cv::Rect templateRect;
bool templateSelected = false;
cv::Point p1, p2;

// Mouse event handler to select the template
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        p1 = cv::Point(x, y);
    } else if (event == cv::EVENT_LBUTTONUP) {
        p2 = cv::Point(x, y);
        templateRect = cv::Rect(p1, p2);
        templateSelected = true;
    }
}

//=============================================== Camera Class ====================================================//

class Camera {
public:
    int camera_index = 0;
    int desired_width = 640;
    int desired_height = 480;
    cv::Mat frame;
    cv::VideoCapture capture;

    Camera() {
        capture.open(this->camera_index);
        if (!capture.isOpened()) {
            throw std::runtime_error("Failed to open camera.");
        }
        capture.set(cv::CAP_PROP_FRAME_WIDTH, this->desired_width);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, this->desired_height);
    }

    ~Camera() {
        if (capture.isOpened()) {
            capture.release();
        }
        cv::destroyAllWindows();
    }

    bool getFrame(cv::Mat& outputFrame) {
        bool success = capture.read(outputFrame);
        if (!success) {
            throw std::runtime_error("Failed to read frames from the camera");
        }
        return success;
    }
};

//=============================================== Pyramid Matching ====================================================//

class PyramidMatching {
public:
    int pyramidLevel;
    std::vector<cv::Mat> templatePyramid;

    PyramidMatching(const cv::Mat& inputFrame, int levels = 3) {
        this->pyramidLevel = levels;
        this->templatePyramid = buildPyramid(inputFrame, this->pyramidLevel);
    }

    ~PyramidMatching() {
        templatePyramid.clear();
    }
    
    cv::Point match(const cv::Mat& frameToMatch) {
    
        std::vector<cv::Mat> framePyramid = buildPyramid(frameToMatch, this->pyramidLevel);
        
        if (framePyramid.size() != this->templatePyramid.size()) {
            throw std::runtime_error("Pyramid sizes do not match for comparison.");
        }
        
        double bestOverallScore = -1.0;
        cv::Point bestOverallLocation(0, 0);

        for (size_t i = 0; i < this->pyramidLevel; ++i) {
            cv::Mat& frameLevel = framePyramid[i];
            cv::Mat& templateLevel = templatePyramid[i];

            if (templateLevel.rows > frameLevel.rows || templateLevel.cols > frameLevel.cols) {
                continue;
            }

            for (int y = 0; y <= frameLevel.rows - templateLevel.rows; y+=4) {
                for (int x = 0; x <= frameLevel.cols - templateLevel.cols; x+=4) {
                    cv::Mat subRegion = getSubRegion(frameLevel, x, y, templateLevel.cols, templateLevel.rows);
                    double similarity = cosineSim(subRegion, templateLevel);

                    if (similarity > bestOverallScore) {
                        bestOverallScore = similarity;
                        // Scale the location back to the original image coordinates
                        double scaleFactor = std::pow(2.0, i);
                        bestOverallLocation.x = static_cast<int>(x * scaleFactor);
                        bestOverallLocation.y = static_cast<int>(y * scaleFactor);
                    }
                }
            }
        }
        
        std::cout << "Best overall score: " << bestOverallScore << " at location: (" 
                  << bestOverallLocation.x << ", " << bestOverallLocation.y << ")" << std::endl;

        return bestOverallLocation;
    }

private:
    cv::Mat getSubRegion(const cv::Mat& source, int x, int y, int width, int height) {
    
        cv::Mat subRegion(height, width, source.type());
        for (int k = 0; k < height; ++k) {
            uchar* subRegionRowPtr = subRegion.ptr<uchar>(k);
            const uchar* sourceRowPtr = source.ptr<uchar>(y + k);

            for (int l = 0; l < width; ++l) {
                subRegionRowPtr[l] = sourceRowPtr[x + l];
            }
        }
        return subRegion;
    }

    std::vector<cv::Mat> buildPyramid(const cv::Mat& inputFrame, int levels) {
        std::vector<cv::Mat> pyramid;
        cv::Mat currentFrame = inputFrame.clone();

        if (currentFrame.empty()) {
            throw std::runtime_error("Input frame for pyramid is empty.");
        }

        for (int i = 0; i < levels; i++) {
            pyramid.push_back(currentFrame.clone());
            if (currentFrame.cols > 1 && currentFrame.rows > 1) {
                 cv::resize(currentFrame, currentFrame, cv::Size(currentFrame.cols / 2, currentFrame.rows / 2));
            } else {
                 break;
            }
        }
        return pyramid;
    }

    double cosineSim(const cv::Mat& mat1, const cv::Mat& mat2) {
        if (mat1.size() != mat2.size() || mat1.type() != mat2.type()) {
            throw std::runtime_error("Images must have the same size and type for cosine similarity.");
        }

        cv::Mat vec1, vec2;
        mat1.reshape(1, 1).convertTo(vec1, CV_64F);
        mat2.reshape(1, 1).convertTo(vec2, CV_64F);
        
        double dotProduct = vec1.dot(vec2);
        double normA = cv::norm(vec1);
        double normB = cv::norm(vec2);

        if (normA == 0 || normB == 0) {
            return 0.0;
        }

        return dotProduct / (normA * normB);
    }
};

//=============================================== Main Program ====================================================//

int main() {
    try {
        Camera cam;
        cv::Mat frame;
        cv::Mat templateImage;

        cv::namedWindow("Live Camera Feed");
        cv::setMouseCallback("Live Camera Feed", onMouse);

        std::cout << "Draw a rectangle on the live feed to select your template." << std::endl;
        std::cout << "Press 's' to set the template, or 'q' to quit." << std::endl;

        // Stage 1: Select the template
        while (true) {
            cam.getFrame(frame);
            if (frame.empty()) {
                std::cerr << "Frame is empty. Exiting." << std::endl;
                return -1;
            }

            if (templateSelected && templateRect.width > 0 && templateRect.height > 0) {
                cv::rectangle(frame, templateRect, cv::Scalar(0, 255, 0), 2);
            }
            
            cv::imshow("Live Camera Feed", frame);
            int key = cv::waitKey(1);
            if (key == 's' && templateSelected) {
                templateImage = frame(templateRect).clone();
                break;
            } else if (key == 'q') {
                return 0;
            }
        }
        
        if (templateImage.empty() || templateRect.width <= 0 || templateRect.height <= 0) {
            std::cerr << "Template was not selected or was invalid. Exiting." << std::endl;
            return -1;
        }

        PyramidMatching pm(templateImage, 3);
        std::cout << "Template selected. Starting real-time matching." << std::endl;
        std::cout << "Press 'q' to quit." << std::endl;

        // Stage 2: Perform real-time matching
        while (true) {
            cam.getFrame(frame);
            if (frame.empty()) {
                break;
            }

            cv::Point bestLocation = pm.match(frame);
            templateRect.x = bestLocation.x;
            templateRect.y = bestLocation.y;
            
            cv::rectangle(frame, templateRect, cv::Scalar(0, 0, 255), 2);
            
            cv::imshow("Live Camera Feed", frame);

            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}