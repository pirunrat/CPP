#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK 16
#define PI 3.14159265

using namespace cv;

// ===== CUDA Kernels =====

__global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ float kernel[5][5];
    if (threadIdx.x < 5 && threadIdx.y < 5) {
        const float G[5][5] = {
            {2, 4, 5, 4, 2},
            {4, 9,12, 9, 4},
            {5,12,15,12, 5},
            {4, 9,12, 9, 4},
            {2, 4, 5, 4, 2}
        };
        kernel[threadIdx.y][threadIdx.x] = G[threadIdx.y][threadIdx.x] / 159.0f;
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 2 && y >= 2 && x < width - 2 && y < height - 2) {
        float val = 0.0;
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int px = x + kx;
                int py = y + ky;
                val += input[py * width + px] * kernel[ky + 2][kx + 2];
            }
        }
        output[y * width + x] = static_cast<unsigned char>(val);
    }
}

__global__ void sobel_kernel(unsigned char* input, float* grad, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int gx = -input[(y-1)*width + (x-1)] - 2*input[y*width + (x-1)] - input[(y+1)*width + (x-1)]
                 + input[(y-1)*width + (x+1)] + 2*input[y*width + (x+1)] + input[(y+1)*width + (x+1)];
        int gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                 + input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];
        grad[y * width + x] = sqrtf(gx * gx + gy * gy);
    }
}

__global__ void threshold_kernel(float* grad, unsigned char* output, int width, int height, float low, float high) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float g = grad[y * width + x];
        if (g >= high) output[y * width + x] = 255;
        else if (g >= low) output[y * width + x] = 128;
        else output[y * width + x] = 0;
    }
}

// ===== Main Function =====

int main() {
    VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_BUFFERSIZE, 1);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera!\n";
        return -1;
    }

    Mat frame, gray;
    cap >> frame;
    resize(frame, frame, Size(640, 480));
    int width = frame.cols;
    int height = frame.rows;
    size_t img_size = width * height * sizeof(unsigned char);
    size_t float_size = width * height * sizeof(float);

    unsigned char *d_input, *d_blur, *d_output;
    float *d_grad;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_blur, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_grad, float_size);

    namedWindow("Canny CUDA", WINDOW_AUTOSIZE);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        cudaMemcpy(d_input, gray.data, img_size, cudaMemcpyHostToDevice);

        dim3 threads(BLOCK, BLOCK);
        dim3 blocks((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);

        gaussian_blur_kernel<<<blocks, threads>>>(d_input, d_blur, width, height);
        sobel_kernel<<<blocks, threads>>>(d_blur, d_grad, width, height);
        threshold_kernel<<<blocks, threads>>>(d_grad, d_output, width, height, 50, 100);
        cudaDeviceSynchronize();

        std::vector<unsigned char> output_data(width * height);
        cudaMemcpy(output_data.data(), d_output, img_size, cudaMemcpyDeviceToHost);

        Mat edge_frame(height, width, CV_8UC1, output_data.data());
        imshow("Canny CUDA", edge_frame);

        if (waitKey(1) == 27) break;  // ESC to exit
    }

    cudaFree(d_input);
    cudaFree(d_blur);
    cudaFree(d_output);
    cudaFree(d_grad);
    return 0;
}
