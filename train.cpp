#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
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

int main() {
    // Initialize the MLP with appropriate sizes
    MLP net(4, 3, {1024, 512, 256}, 0.001f); // Changed input size to 4 for the additional alpha channel

    // Load the weights if available
    try {
        net.load_weights("weights.bin");
    } catch (const std::exception &e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
    }

    // Load the broken image
    int width, height;
    std::vector<glm::vec3> broken_image = load_image("broken.bmp", width, height);

    // Set the batch size
    const size_t batch_size = 32; // Adjust based on your needs

    // Prepare for training
    std::vector<glm::vec4> inputs;
    std::vector<glm::vec4> targets;

    for (unsigned int i = 0; i < broken_image.size(); ++i) {
        if (broken_image[i].x < 0.1f) { // Assuming broken pixels have a low value
            glm::vec4 input;
            if (i > 0) {
                input = glm::vec4(broken_image[i - 1].r, broken_image[i - 1].g, broken_image[i - 1].b, 1.0f); // Previous pixel as input
            } else {
                input = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); // Default input if first pixel
            }

            // Prepare the input and target
            inputs.push_back(input);
            targets.push_back(glm::vec4(broken_image[i].r, broken_image[i].g, broken_image[i].b, 1.0f)); // Original pixel value as target

            // Train in batches
            if (inputs.size() == batch_size || i == broken_image.size() - 1) {
                net.train(inputs, targets); // Train on the collected batch
                inputs.clear(); // Clear for next batch
                targets.clear();
            }
        }
    }

    // Save the weights after training
    net.save_weights("weights.bin");
    std::cout << "Training complete, weights saved as 'weights.bin'.\n";

    return 0;
}
