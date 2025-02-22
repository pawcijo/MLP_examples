#include <iostream>
#include <vector>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <epochs>" << std::endl;
        return 1;
    }

    // Parse the number of epochs from the command-line argument
    int epochs = std::stoi(argv[1]);

    // Load and preprocess image
    std::string image_filename = "image.bmp";
    int width, height;
    std::vector<float> inputs = load_image(image_filename, width, height);

    // Define the MLP architecture based on image size
    std::vector<int> layers = {width * height, 128, 64, 32, 64, 128, width * height}; // Example architecture
    float learning_rate = 0.01;

    // Create MLP instance
    MLP mlp(layers, learning_rate);

    // Define labels (for simplicity, let's assume labels are the same as inputs for reconstruction)
    std::vector<float> labels = inputs; // Ensure labels are also flat

    // Create vectors of vectors for inputs and labels
    std::vector<std::vector<float>> input_batch = {inputs}; // Wrap input in a vector of vectors
    std::vector<std::vector<float>> label_batch = {labels}; // Wrap label in a vector of vectors

    // Train the MLP
    mlp.train(input_batch, label_batch, epochs); // Use the wrapped vectors

    // Save the trained model
    std::string model_filename = "mlp_model.dat";
    mlp.save_model(model_filename);

    return 0;
}
