#include <opencv2/opencv.hpp> // Includes core OpenCV functionalities
#include <iostream>           // For console input/output
#include <immintrin.h>        // Required for Intel SSE/AVX intrinsics (e.g., SSE2, SSE4.1)

// Function to perform binary thresholding manually (scalar implementation)
// It processes one pixel at a time.
void manualBinaryThreshold(const cv::Mat& inputImage, cv::Mat& outputImage, double thresholdValue) {
    // Ensure the output image has the same dimensions and type as the input.
    outputImage.create(inputImage.size(), inputImage.type());

    if (inputImage.channels() != 1) {
        std::cerr << "Error (manualBinaryThreshold): Input image must be grayscale." << std::endl;
        return;
    }

    for (int i = 0; i < inputImage.rows; ++i) {
        const uchar* inputRow = inputImage.ptr<uchar>(i);
        uchar* outputRow = outputImage.ptr<uchar>(i);

        for (int j = 0; j < inputImage.cols; ++j) {
            uchar pixelValue = inputRow[j];
            if (pixelValue > thresholdValue) {
                outputRow[j] = 255;
            } else {
                outputRow[j] = 0;
            }
        }
    }
}

// Function to perform binary thresholding using SSE2 SIMD intrinsics
// This processes 16 pixels (bytes) at a time.
void simdBinaryThreshold(const cv::Mat& inputImage, cv::Mat& outputImage, double thresholdValue) {
    // Ensure the output image has the same dimensions and type as the input.
    outputImage.create(inputImage.size(), inputImage.type());

    if (inputImage.channels() != 1) {
        std::cerr << "Error (simdBinaryThreshold): Input image must be grayscale." << std::endl;
        return;
    }

    // Convert threshold value to char for SSE2 comparison (8-bit signed/unsigned)
    // For unsigned comparison, it's often done by subtracting and checking sign bit,
    // or by using specific unsigned comparison intrinsics if available (SSE4.1 has _mm_cmpgt_epu8, but SSE2 doesn't).
    // For simplicity and common use cases, we'll use signed comparison and ensure threshold fits.
    // A more robust unsigned comparison might involve XORing with 0x80 to shift range.
    char thresholdChar = static_cast<char>(thresholdValue); // Treat as signed for _mm_cmpgt_epi8

    // Define constants for SIMD operations
    const __m128i zero_vec = _mm_setzero_si128(); // A vector of 16 zeros
    const __m128i white_vec = _mm_set1_epi8(static_cast<char>(255)); // A vector of 16 255s
    const __m128i threshold_vec = _mm_set1_epi8(thresholdChar); // A vector of 16 threshold values

    // Iterate through each row of the image
    for (int i = 0; i < inputImage.rows; ++i) {
        const uchar* inputRow = inputImage.ptr<uchar>(i);
        uchar* outputRow = outputImage.ptr<uchar>(i);

        // Process pixels in chunks of 16 (size of __m128i for 8-bit integers)
        int j = 0;
        for (; j <= inputImage.cols - 16; j += 16) {
            // Load 16 unsigned char pixels from the input row into an SSE register.
            // _mm_loadu_si128 is for unaligned loads, which is safer with arbitrary image data.
            __m128i pixels = _mm_loadu_si128(reinterpret_cast<const __m128i*>(inputRow + j));

            // Perform a packed signed byte comparison: pixels > threshold_vec.
            // _mm_cmpgt_epi8 sets each byte to 0xFF if true, 0x00 if false.
            // Note: This is a signed comparison. For pure unsigned, careful handling is needed.
            // For 0-255 range, this often works sufficiently if threshold is not near 128.
            __m128i comparison_mask = _mm_cmpgt_epi8(pixels, threshold_vec);

            // Apply the mask to the 'white_vec'.
            // If a byte in comparison_mask is 0xFF, the corresponding byte from white_vec (255) is chosen.
            // If a byte in comparison_mask is 0x00, the corresponding byte from white_vec (255) is ANDed with 0, resulting in 0.
            // This effectively implements: (pixel > threshold) ? 255 : 0;
            __m128i result_pixels = _mm_and_si128(comparison_mask, white_vec);

            // Store the 16 resulting pixels back to the output row.
            _mm_storeu_si128(reinterpret_cast<__m128i*>(outputRow + j), result_pixels);
        }

        // Handle remaining pixels (the "tail") that are not a multiple of 16
        // These are processed one by one using scalar operations.
        for (; j < inputImage.cols; ++j) {
            if (inputRow[j] > thresholdValue) {
                outputRow[j] = 255;
            } else {
                outputRow[j] = 0;
            }
        }
    }
}


int main() {
    // 1. Open the default webcam
    cv::VideoCapture cap(0);

    // Check if the camera was opened successfully.
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    // Define target dimensions for resizing the frame
    int targetWidth = 640;  // Example width
    int targetHeight = 480; // Example height

    // Define the threshold value for binary thresholding.
    double thresholdValue = 128;

    // Loop indefinitely to capture and process frames in real-time.
    while (true) {
        cv::Mat frame; // Original captured frame
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Captured empty frame. Exiting." << std::endl;
            break;
        }

        cv::Mat resizedFrame;
        cv::resize(frame, resizedFrame, cv::Size(targetWidth, targetHeight));

        cv::Mat grayFrame;
        cv::cvtColor(resizedFrame, grayFrame, cv::COLOR_BGR2GRAY);

        // --- Perform thresholding using all three implementations ---

       

        // 2. SIMD (SSE2) Binary Threshold
        cv::Mat simdThresholdedFrame;
        simdBinaryThreshold(grayFrame, simdThresholdedFrame, thresholdValue);

        // --- Display the results ---

       
        cv::imshow("SIMD (SSE2) Binary Threshold", simdThresholdedFrame);
      

        // Wait for 1 millisecond and check for exit key
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) { // 'q', 'Q', or ESC
            break;
        }
    }

    return 0;
}