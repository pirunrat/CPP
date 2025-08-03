#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstdint>


std::vector<std::vector<std::vector<uint8_t>>> get_subsample(cv::Mat &image) {

    int patch_size = 40;
    int step_size = 8;
    
    std::vector<std::vector<std::vector<uint8_t>>> out;

    int rows = image.rows;
    int cols = image.cols;

    // Iterate through the image to find the top-left corner of each patch.
    for (int i = 0; i <= rows - patch_size; i += step_size) {
        for (int j = 0; j <= cols - patch_size; j += step_size) {
            
            // Create a temporary vector to hold the current patch's pixels.
            std::vector<std::vector<uint8_t>> current_patch;
            current_patch.resize(patch_size);

            // Iterate through the patch's dimensions to copy the data.
            for (int k = 0; k < patch_size; ++k) {
                current_patch[k].resize(patch_size);
                for (int l = 0; l < patch_size; ++l) {
                    // Access the pixel in the original image and assign it.
                    int original_row = i + k;
                    int original_col = j + l;
                    if (original_row < rows && original_col < cols) {
                        current_patch[k][l] = image.ptr<uint8_t>(original_row)[original_col];
                    }
                }
            }
            // Push the completed patch into the output vector.
            out.push_back(current_patch);
        }
    }

    return out;
}





int main(){
    std::string image_path = "../Test/download.jpg";

    cv::Mat img = cv::imread(image_path);

    // Check if the image was loaded successfully
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    std::vector<std::vector<std::vector<uint8_t>>> out = get_subsample(img);

    // Print the shape of the 3D vector
    if (!out.empty()) {
        int num_patches = out.size();
        int patch_rows = out[0].size();
        int patch_cols = out[0][0].size();

        std::cout << "Shape of the output: (" << num_patches << ", " << patch_rows << ", " << patch_cols << ")" << std::endl;
    } else {
        std::cout << "The output vector is empty." << std::endl;
    }

    return 0;
}


