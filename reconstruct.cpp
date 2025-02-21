#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include "src/Mlp.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#pragma clang diagnostic pop

std::vector<glm::vec3> load_image(const std::string &filename, int &width, int &height) {
    unsigned char *img = stbi_load(filename.c_str(), &width, &height, nullptr, 3); // Load as 24-bit RGB
    std::vector<glm::vec3> data;

    if (img) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = (y * width + x) * 3; // 3 channels (R, G, B)
                float r = img[index] / 255.0f;
                float g = img[index + 1] / 255.0f;
                float b = img[index + 2] / 255.0f;
                data.emplace_back(r, g, b); // Store as vec3
            }
        }
        stbi_image_free(img); // Free the image memory
    } else {
        std::cerr << "Failed to load image: " << filename << std::endl;
    }

    return data;
}

void save_image(const std::string &filename, const std::vector<glm::vec3> &data, int width, int height) {
    std::vector<unsigned char> img_data(width * height * 3); // 3 channels for RGB
    for (int i = 0; i < width * height; ++i) {
        img_data[i * 3] = static_cast<unsigned char>(glm::clamp(data[i].x * 255.0f, 0.0f, 255.0f));
        img_data[i * 3 + 1] = static_cast<unsigned char>(glm::clamp(data[i].y * 255.0f, 0.0f, 255.0f));
        img_data[i * 3 + 2] = static_cast<unsigned char>(glm::clamp(data[i].z * 255.0f, 0.0f, 255.0f));
    }
    stbi_write_bmp(filename.c_str(), width, height, 3, img_data.data()); // Save as BMP
}

int main() {
    // Initialize the MLP with appropriate sizes (input_size, output_size, hidden_layers, learning_rate)
    MLP net(4, 3, {1024, 512, 256}, 0.001f); // Adjust input size to 4 for the alpha channel
    try {
        net.load_weights("weights.bin");
    } catch (const std::exception &e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        return -1; // Exit if loading weights fails
    }

    // Load the broken image
    int width, height;
    std::vector<glm::vec3> broken_image = load_image("broken.bmp", width, height);

    // Reconstruct the broken image
    for (unsigned int i = 0; i < broken_image.size(); ++i) {
        if (broken_image[i].x < 0.1f) { // Assuming broken pixels have a low value
            glm::vec4 input;
            if (i > 0) {
                input = glm::vec4(broken_image[i - 1].r, broken_image[i - 1].g, broken_image[i - 1].b, 1.0f); // Previous pixel as input
            } else {
                input = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); // Default input if first pixel
            }
            glm::vec4 reconstructed_pixel = net.feedforward(input); // Predict the missing pixel value
            broken_image[i] = glm::vec3(reconstructed_pixel.x, reconstructed_pixel.y, reconstructed_pixel.z); // Store the reconstructed pixel
        }
    }

    // Save the reconstructed image
    save_image("reconstructed_image.bmp", broken_image, width, height);
    std::cout << "Reconstruction complete, saved as 'reconstructed_image.bmp'.\n";

    return 0;
}
