#include <iostream>
#include <vector>
#include <string>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#pragma clang diagnostic pop

#include "MLP.hpp"

// Function to load image from a file using STB Image
std::vector<float> load_image(const std::string& filename, int& width, int& height) {
    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 1); // Load image as grayscale

    if (data == nullptr) {
        std::cerr << "Error: Could not load image " << filename << std::endl;
        exit(1);
    }

    std::vector<float> image(width * height); // Create a flat vector for grayscale image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image[y * width + x] = data[y * width + x] / 255.0f; // Normalize pixel values to [0, 1]
        }
    }

    stbi_image_free(data);
    return image; // Return the flat vector
}

// Function to save the reconstructed image
void save_image(const std::string& filename, const std::vector<float>& image, int width, int height) {
    std::vector<unsigned char> output_image(width * height);
    for (int i = 0; i < width * height; ++i) {
        output_image[i] = static_cast<unsigned char>(image[i] * 255.0f); // Convert back to [0, 255]
    }

    stbi_write_bmp(filename.c_str(), width, height, 1, output_image.data()); // Save as grayscale BMP
}

int main() {
    // Load the broken image
    std::string broken_image_filename = "broken.bmp";
    int width, height;
    std::vector<float> inputs = load_image(broken_image_filename, width, height);

    // Define the MLP architecture based on image size
    std::vector<int> layers = {width * height, 128, 64, 32, 64, 128, width * height}; // Example architecture
    float learning_rate = 0.01;

    // Create MLP instance
    MLP mlp(layers, learning_rate);

    // Load the trained model
    std::string model_filename = "mlp_model.dat";
    mlp.load_model(model_filename);

    // Reconstruct the image using the MLP
    std::vector<float> reconstructed_image = mlp.predict(inputs);

    // Save the reconstructed image
    std::string reconstructed_image_filename = "reconstructed.bmp";
    save_image(reconstructed_image_filename, reconstructed_image, width, height);

    std::cout << "Reconstructed image saved as " << reconstructed_image_filename << std::endl;

    return 0;
}
