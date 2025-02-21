#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>

class MLP {
private:
    std::vector<glm::mat4> weights; // Matrix for weights (up to 4x4; can be expanded to larger sizes)
    std::vector<glm::vec4> biases;  // Bias vector for each layer
    float learning_rate;
    int input_size, output_size;
    std::vector<int> hidden_layers; // Number of neurons per hidden layer
    int previous_size; // Size of the previous layer

    glm::vec4 activate(const glm::vec4 &x);
    glm::vec4 activate_derivative(const glm::vec4 &x);
    glm::vec4 feedforward_internal(const glm::vec4 &input);

public:
    MLP(int input_size = 12288, int output_size = 12288, std::vector<int> hidden_layers = {1024, 512, 256}, float lr = 0.001f);
    
    glm::vec4 feedforward(const glm::vec4 &input);
    void train(const std::vector<glm::vec4> &inputs, const std::vector<glm::vec4> &targets, int epochs = 1);
    
    void load_weights(const std::string &filename);
    void save_weights(const std::string &filename);
}; 
